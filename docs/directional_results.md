# Directional (Circular) Analysis of Detected Shots

*Generated 2026-07-29 from `events_clip1/2/3.json` (n = 16 bat-contact events with a
valid ball trajectory). Reproduce with `scratchpad/directional_analysis.py` (basic) and
`scratchpad/directional_extended.py` (extended). Machine-readable outputs:
`docs/figures/directional_stats.json`, `docs/figures/directional_stats_extended.json`.*

## 1. Method

Each detected shot yields a wagon-wheel angle θ ∈ [0°, 360°), where 0° is a straight
drive back toward the bowler and the angle increases clockwise (right-handed batsman
convention). These are **circular** data, so ordinary means and standard deviations are
invalid (they cannot wrap around 360°). We use circular statistics (Mardia & Jupp,
2000; Fisher, 1993; Berens, 2009):

- **Mean resultant vector.** With C = Σcos θᵢ, S = Σsin θᵢ, the mean direction is
  θ̄ = atan2(S, C) and the mean resultant length R̄ = √(C²+S²)/n ∈ [0, 1]
  (0 = fully dispersed, 1 = identical directions).
- **Dispersion:** circular variance V = 1 − R̄, circular standard deviation
  s = √(−2 ln R̄), and Batschelet angular deviation s₀ = √(2(1−R̄)), all in degrees.
- **von Mises concentration κ** (maximum-likelihood, with the Best–Fisher small-sample
  correction), the circular analogue of 1/σ².
- **Uniformity tests** (H₀: no preferred direction). We report **four** complementary
  tests because they have different sensitivities: the **Rayleigh** test (most powerful
  against a single mode), **Rao's spacing** test (sensitive to multimodal / patchy
  departures), and the **Kuiper** and **Watson U²** tests (omnibus EDF tests). p-values
  are obtained by Monte-Carlo (20 000 uniform samples) for exactness at small n.
- **Axial (bimodality) test:** Rayleigh test on doubled angles 2θ, which detects two
  modes 180° apart.
- **Circular–linear correlation** between shot direction and carry distance (Mardia's
  r_xl), with a 20 000-permutation p-value.
- **Multi-sample comparison** across the three clips: **Watson–Williams** F-test
  (H₀: equal mean directions).
- **95% confidence intervals** for θ̄ and R̄ from 10 000 nonparametric bootstrap
  resamples.

## 2. Location and concentration (pooled, n = 16)

| Quantity | Value | 95% CI |
| --- | --- | --- |
| Mean direction θ̄ | 229.0° | [175.0°, 268.5°] (bootstrap) |
| Mean resultant R̄ | 0.434 | [0.231, 0.707] (bootstrap) |
| von Mises κ (MLE) | 0.964 | — |
| Circular variance V | 0.566 | — |
| Circular SD s | 74.0° | — |
| Angular deviation s₀ | 60.9° | — |

The mean direction points to **229° — the leg-side / square-leg region**. However κ ≈ 1
and the wide bootstrap CI (spanning ~94°) show the directions are **only moderately
concentrated**, as expected for n = 16.

## 3. Tests of uniformity

| Test | Statistic | p (Monte-Carlo) | Rejects H₀ at 0.05? |
| --- | --- | --- | --- |
| Rayleigh | Z = 3.019 | 0.048 | marginally yes |
| Rao's spacing | U = 147.0 | 0.191 | no |
| Kuiper | V = 1.663 | 0.077 | no |
| Watson U² | 0.178 | 0.059 | no |

*(Rayleigh analytic p = 0.046, agreeing with its Monte-Carlo value.)*

**Interpretation — report this honestly.** The evidence for a preferred scoring
direction is **suggestive but not robust**. The Rayleigh test — which is the most
powerful test when the alternative is a single dominant direction — is *marginally*
significant (p ≈ 0.048), consistent with the visible leg-side concentration. But the
three more general tests do **not** reject uniformity at α = 0.05 (Kuiper and Watson U²
sit just above threshold; Rao's spacing is clearly non-significant). At n = 16 this is
the expected signature of a **weak, unimodal trend rather than a strong bias**. The
finding should be stated as *a leg-side tendency detectable by the Rayleigh test, not
yet corroborated by omnibus tests at this sample size.*

## 4. Modality, correlation, and field-zone results

- **Bimodality (axial test):** principal axis 104.9°/284.9°, axial R̄ = 0.246,
  p = 0.390 → **no significant two-lobed structure**; the distribution is best described
  as a single weak mode, not off-side-vs-leg-side symmetry.
- **Direction vs carry distance:** Mardia r_xl = 0.473, permutation p = 0.186 →
  **no significant association** between where a shot is played and how far it carries.
- **Field-zone tally:** Leg side 10 (62.5%), off side 3 (18.8%), behind point/edge 3
  (18.8%), straight 0. A binomial test of the leg-side share against 0.5 is **not
  significant** (p = 0.227) — again, a tendency rather than a proven bias.

## 5. Inter-clip comparison

Watson–Williams F(2, 13) = 2.55, p = 0.116 (pooled κ = 1.38 > 1, so the test's
assumptions hold). The three clips do **not** differ significantly in mean direction
(Clip 1: 227°, Clip 2: 183°, Clip 3: 271°), which **justifies pooling** them for the
analyses above.

## 6. Per-shot summary

| Shot | n | Mean direction | Mean carry |
| --- | --- | --- | --- |
| Flick | 4 | 272.7° | 5.9 m |
| Pull Shot | 3 | 268.8° | 8.4 m |
| Edge | 3 | 169.9° | 8.0 m |
| Leg Glance | 2 | 217.8° | 6.6 m |
| Lofted Drive | 1 | 94.3° | 19.7 m |
| Late Cut | 1 | 148.4° | 10.1 m |
| On Drive | 1 | 322.0° | 2.5 m |
| Square Cut | 1 | 97.5° | 7.3 m |

The most frequent strokes (Flick, Pull) cluster tightly on the leg side (~270°), which
is what drives the pooled mean direction. The single Lofted Drive gave by far the
longest carry (19.7 m), as expected for an aerial stroke.

## 7. Conclusion for the paper

Across 16 automatically detected shots the batsmen show a **leg-side directional
tendency** (mean direction 229°, Rayleigh p ≈ 0.048), but this is **not yet corroborated
by omnibus uniformity tests, the leg-side proportion test, or a narrow confidence
interval**, all of which are limited by the small automatically-extracted sample. There
is no significant multimodality, no direction–distance coupling, and no inter-clip
difference in mean direction. The result is best presented as a **demonstration that the
pipeline supports quantitative circular analysis of batting direction**, with the
directional bias itself flagged as preliminary pending a larger, manually verified event
set.

*Caveat: events were extracted automatically and inherit the pipeline's known
limitations (sparse ball detection ~9%, occasional follow-through/replay mis-fires). The
angle for each event is derived from the post-contact ball trajectory and is independent
of the delivery-segmentation logic.*

## Figures

![Basic directional analysis](figures/wagon_wheel_directional.png)

**Fig. X.** Wagon-wheel directional analysis of the n = 16 detected shots. **(a)** shot
direction (angle) vs predicted carry distance (radius, m), coloured by clip; **(b)**
12-bin circular histogram with the mean-direction arrow (length ∝ R̄).

![Extended directional analysis](figures/directional_extended.png)

**Fig. Y.** **(a)** Fitted von Mises distribution (κ = 0.96) over the observed histogram,
with the mean direction (229°) and its 95% CI wedge — the near-circular density curve
reflects the low concentration. **(b)** Per-clip mean-direction vectors and individual
shots; the Watson–Williams test (F = 2.55, p = 0.116) finds no significant difference,
justifying pooling.
