"""Embedding backbones for instance-level retrieval.

DINOv2 is the default rather than CLIP because this is an *instance* problem --
"which SKU is this" -- and self-supervised DINOv2 features retain instance
identity, where CLIP's language-aligned features tend to collapse visually
distinct SKUs into a shared semantic bucket ("a white sneaker"). That claim is
tested in the ablation ladder rather than assumed: `build_backbone("clip-...")`
is available so the comparison can actually be run.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn.functional as F
from PIL import Image

# Short names -> (timm model id, native resolution, resizable?).
#
# DINOv2 vs SigLIP2 is an OPEN QUESTION for this domain, not a settled one:
# DINOv2 wins instance retrieval on landmarks (ROxford/RParis), but FORB and
# ILIAS both report CLIP/SigLIP ahead on *product* retrieval, where identity is
# carried by brand marks and colourway rather than rigid 3D texture. Sneakers
# look more like the product case. So the registry carries both and the
# bake-off decides -- see `scripts/bakeoff.py`.
BACKBONES = {
    # name              timm id                                      native  resizable
    "dinov2-base":     ("vit_base_patch14_dinov2.lvd142m",             518,   True),
    "dinov2-small":    ("vit_small_patch14_dinov2.lvd142m",            518,   True),
    "dinov2-base-reg": ("vit_base_patch14_reg4_dinov2.lvd142m",        518,   True),
    "dinov2-small-reg":("vit_small_patch14_reg4_dinov2.lvd142m",       518,   True),
    "siglip2-base":    ("vit_base_patch16_siglip_256.v2_webli",        256,   False),
    "siglip2-base-384":("vit_base_patch16_siglip_384.v2_webli",        384,   False),
    "clip-base":       ("vit_base_patch16_clip_224.openai",            224,   False),
}


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EncodeConfig:
    """How an image becomes a vector.

    pooling:
      cls        CLS token only
      cls_patch  CLS concatenated with mean-pooled patch tokens (often stronger
                 for retrieval -- patch means carry local texture the CLS drops)
    """

    image_size: int = 224
    pooling: str = "cls_patch"
    flip_tta: bool = False


class Backbone:
    def __init__(
        self,
        name: str = "dinov2-base",
        device: str | None = None,
        config: EncodeConfig | None = None,
    ):
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
        self.name = name
        self.cfg = config or EncodeConfig()
        self.device = pick_device(device)

        timm_id, native, resizable = BACKBONES[name]
        if not resizable:
            # SigLIP/CLIP towers use learned absolute position embeddings tied to
            # their training grid; forcing another size degrades them for no gain.
            self.cfg.image_size = native
        kwargs = {"pretrained": True, "num_classes": 0}
        if resizable:
            kwargs["img_size"] = self.cfg.image_size
        self.model = timm.create_model(timm_id, **kwargs)
        self.model.eval().to(self.device)

        data_cfg = timm.data.resolve_model_data_config(self.model)
        # Force our own size: resolve_model_data_config reports the checkpoint's
        # native resolution (518 for DINOv2), which is far more compute than
        # retrieval needs.
        data_cfg["input_size"] = (3, self.cfg.image_size, self.cfg.image_size)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)

    @property
    def dim(self) -> int:
        d = self.model.num_features
        return d * 2 if self.cfg.pooling == "cls_patch" else d

    def _global(self, tokens: torch.Tensor) -> torch.Tensor:
        """The model's own global descriptor.

        DINOv2 exposes a CLS token at position 0. SigLIP has no CLS at all
        (`num_prefix_tokens == 0`) and pools with a learned attention head, so
        taking token 0 would silently grab an arbitrary patch. Defer to
        `forward_head` in that case, which applies the trained pooler.
        """
        if self.n_prefix > 0:
            return tokens[:, 0]
        return self.model.forward_head(tokens, pre_logits=True)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        glob = self._global(tokens)
        if self.cfg.pooling == "cls":
            return glob
        patches = tokens[:, self.n_prefix :].mean(dim=1)
        return torch.cat([glob, patches], dim=-1)

    @torch.inference_mode()
    def encode_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed float batch -> L2-normalised vectors."""
        batch = batch.to(self.device, non_blocking=True)
        feats = self._pool(self.model.forward_features(batch))
        if self.cfg.flip_tta:
            flipped = self._pool(self.model.forward_features(torch.flip(batch, dims=[3])))
            # Max-pool over the original and its mirror. Catalogue shots are
            # near-always one shoe in one orientation, so roughly half of real
            # query photos show the opposite foot.
            feats = torch.maximum(F.normalize(feats, dim=-1), F.normalize(flipped, dim=-1))
        return F.normalize(feats, dim=-1).float().cpu()

    @torch.inference_mode()
    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """Raw patch tokens, (B, n_patches, D) -- prefix tokens stripped.

        Used for saliency localisation and for dense patch-correspondence
        re-ranking, both of which need spatial structure the pooled vector drops.
        """
        batch = batch.to(self.device, non_blocking=True)
        feats = self.model.forward_features(batch)
        return feats[:, self.n_prefix :].float().cpu()

    def grid(self) -> int:
        """Patch grid side length for the configured input size."""
        return self.cfg.image_size // self.model.patch_embed.patch_size[0]

    def encode_images(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            batch = torch.stack([self.transform(im.convert("RGB")) for im in chunk])
            out.append(self.encode_tensor(batch))
        return torch.cat(out) if out else torch.empty(0, self.dim)


def build_backbone(name: str = "dinov2-base", **kw) -> Backbone:
    return Backbone(name=name, **kw)
