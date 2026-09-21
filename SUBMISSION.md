# Submission checklist

Status of every item the brief and the assessment criteria ask for.
**`[ ]` items are not done.** Do not submit until they are.

## Gate — "does it run from a clean checkout using only your README?"

- [x] `README.md` with exact reproduction commands, clean checkout → report
- [x] `scripts/run_all.sh` — one command, resumable
- [x] `requirements.lock` pinned from a working environment
- [x] Write-up present — `REPORT.md`
- [x] Logs present — `logs/SESSION.md` + raw runs in `logs/runs/`
- [x] Test suite — 29 invariant tests including the contamination guard
- [ ] **Verified on a genuinely clean clone**, in a fresh virtualenv, on a
      machine that has never run this code. See "Before you send" below.

## Part A — the matcher

- [x] Catalogue ≥ 5,000 images, self-scraped — **36,506 items / 181,447 images**
- [x] Source choice stated and defended (five sources probed and rejected)
- [x] Ranked top-5 with a confidence score
- [x] Handling of photos with no match — calibrated open-set scorer + conformal
      refusal, where FRR is a design parameter
- [x] Single-lookup latency, measured and broken down by stage

## Part B — the stumper

- [ ] **≥100 phone photographs, shot by you, of items in the catalogue**
- [ ] **Each labelled with the correct catalogue item and the failure conditions**
- [x] Evaluation harness reporting accuracy broken down by failure condition
- [x] Manifest schema + validator + item-disjoint split guard
- [x] Nine corruption operators at five severities
- [x] Statistics: cluster bootstrap, Wilson, McNemar, marginal effects, AURC

**Part B is the blocking item.** The harness, schema and statistics are built and
tested; the photographs are not taken. A synthetic stand-in exercises the
identical code path so the pipeline runs today, but it is labelled as synthetic
everywhere and is not a Part B result.

## Extensions attempted

- [x] **Correct refusal** — conformal prediction sets, empty set = refusal,
      coverage verified against nominal at α ∈ {0.05, 0.10, 0.20}
- [x] **Automated stumper** — corruption generator with realistic condition
      co-occurrence
- [~] **100k / <100 ms CPU** — latency measured and the budget shown to close
      with ViT-S; the 100k index itself is not built
- [ ] Multi-item photos — **cut deliberately** (see DECISIONS.md D14)

## Before you send

```bash
# in a throwaway directory, simulating the reviewer
git clone <your repo> /tmp/gate-check && cd /tmp/gate-check
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q                      # must be green
./scripts/run_all.sh           # must reach reports/eval/report.md
```

If that fails, the reviewer stops reading. They say so explicitly.

## Things to be able to answer in the 45-minute call

They will ask you to change this system live under a constraint you have not
seen. Every one of these is a decision in the repo you should be able to defend
or argue against:

1. Why SigLIP 2 and not DINOv2 — and what measurement would change your mind.
2. Why whitening is kept when it only moves Recall@1 by 4.5 pp.
3. Why views are capped at 4 per item.
4. Why holding out individual items did not produce open-set negatives.
5. Why αQE is built but switched off.
6. Why the bootstrap resamples items rather than photos.
7. What the empty conformal prediction set means and why FRR ≈ α.
8. Where the system is resolution-limited rather than model-limited.

Re-derive at least the first two yourself before submitting. If you disagree
with any of them, change it and record that in `logs/SESSION.md` — a candidate
overruling the tool with evidence is exactly what the criteria reward.
