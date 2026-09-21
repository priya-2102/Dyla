## Results — split `test`

140 photos: 90 hard, 30 clean, 20 out-of-catalogue

### Headline

| set | R@1 SKU | R@1 style | R@5 SKU | median rank of truth | median latency |
|---|---|---|---|---|---|
| clean | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1 | 159 ms |
| hard | 0.556 [0.467, 0.644] | 0.567 [0.478, 0.656] | 0.656 [0.567, 0.744] | 1 | 160 ms |

95% intervals are cluster bootstrap resampling **items**, not photos.

### The gap between the two halves (paired by item, McNemar exact)

- clean-right/hard-wrong: **40**, clean-wrong/hard-right: **0**
- accuracy drop: **+44.4 pp** over 90 pairs, p = 1.819e-12

### Accuracy by failure condition

| condition | n | R@1 SKU | 95% Wilson | R@1 style |
|---|---|---|---|---|
| cluttered_background | 23 | 0.000 | [-0.000, 0.143] | 0.000 |
| small_in_frame | 12 | 0.333 | [0.138, 0.609] | 0.417 |
| low_light | 17 | 0.412 | [0.216, 0.640] | 0.412 |
| partial_occlusion | 10 | 0.500 | [0.237, 0.763] | 0.500 |
| off_angle | 15 | 0.533 | [0.301, 0.752] | 0.533 |
| motion_blur | 21 | 0.667 | [0.454, 0.828] | 0.667 |
| defocus | 22 | 0.682 | [0.473, 0.836] | 0.682 |
| specular_reflection | 14 | 0.786 | [0.524, 0.924] | 0.786 |

> Cells are small. At n=10 a 95% interval spans roughly ±25 pp, so the **ordering** of these conditions is not resolvable from this set. The marginal-effects table and the synthetic dose-response sweep are what the condition ranking should be read from.

### Marginal effect of each condition (exploratory)

| condition | n | log-odds | 95% CI | avg marginal effect |
|---|---|---|---|---|
| cluttered_background | 23 | -2.94 | [-3.34, -2.45] | -58.1 pp |
| low_light | 17 | -1.21 | [-2.05, -0.26] | -22.2 pp |
| partial_occlusion | 10 | -0.32 | [-1.46, +0.63] | -5.6 pp |
| small_in_frame | 12 | -0.24 | [-0.84, +0.37] | -4.2 pp |
| defocus | 22 | +0.13 | [-0.42, +0.70] | +2.3 pp |
| motion_blur | 21 | +0.28 | [-0.46, +1.10] | +4.9 pp |
| specular_reflection | 14 | +0.32 | [-0.38, +1.24] | +5.5 pp |
| off_angle | 15 | +0.44 | [-0.13, +1.01] | +7.4 pp |

> Conditions co-occur, so per-condition accuracy above is confounded. This fits correctness on the condition indicator matrix to estimate each condition's marginal contribution. Penalised and bootstrapped by item; **exploratory** at this sample size.

### Dose response

Each additional adverse condition multiplies the odds of a correct match by **0.366**.

| # conditions | n | accuracy | 95% Wilson |
|---|---|---|---|
| 1 | 55 | 0.709 | [0.579, 0.812] |
| 2 | 26 | 0.308 | [0.165, 0.500] |
| 3 | 9 | 0.333 | [0.121, 0.646] |

### Refusal

- AUROC in-catalogue vs out-of-catalogue: **0.726** (120 in, 20 out)

| outcome | count / rate |
|---|---|
| in-catalogue, accepted & correct | 77 |
| in-catalogue, accepted & **wrong** | 12 |
| in-catalogue, refused (FRR) | 31 (0.258) |
| out-of-catalogue, accepted (**FAR**) | 0.550 |
| **wrong-accept rate** | 0.100 |

> The wrong-accept rate is the error a user actually feels, and a bare FAR/FRR pair hides it.

- AURC **0.147**, excess over oracle (E-AURC) **0.027** — E-AURC isolates how well confidence *ranks* its own errors from how many there are.
