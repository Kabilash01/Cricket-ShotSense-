# CricSense: A Multi-Trigger Vision Pipeline for Automated Bat–Ball Contact Detection and Cricket Shot Classification

> Draft Methodology and Results & Discussion sections for an IEEE journal submission.
> All equations and constants correspond to the released implementation (`src/test3.py`,
> `src/detection/`, `src/tracking/`). Notation is introduced once and reused throughout.

---

## III. METHODOLOGY

### A. System Overview

The proposed pipeline, *CricSense*, ingests a broadcast cricket video stream
and produces, for every delivery, a structured record of the bat–ball contact
event: the contact frame, the post-contact ball trajectory, the wagon-wheel
shot direction, the shot class, and an estimated stroke distance. The system is
organised as a cascade of seven stages operating on a per-frame basis, followed
by a per-event analysis stage triggered at contact:

1. **Ball detection** — a YOLOv8 detector localises the ball.
2. **Ball tracking** — a constant-velocity Kalman filter maintains the ball
   state under intermittent detection.
3. **Data association** — nearest-neighbour gating links detections to the track.
4. **Batsman pose estimation** — a YOLOv8-pose network extracts 17 keypoints of
   the on-strike batsman.
5. **Bat detection** — a YOLOv8 detector localises the bat.
6. **Bat–ball contact detection** — three parallel triggers (bat-box proximity,
   trajectory curvature, wrist velocity) fire a contact event.
7. **Shot analysis** — post-contact trajectory reconstruction, wagon-wheel angle
   computation, and hybrid geometric/deep shot classification.

Let a video be a sequence of frames $\{I_k\}_{k=1}^{K}$ of resolution
$W \times H$ captured at frame rate $f$ (here $W{=}1920$, $H{=}1080$,
$f{=}29.97$ Hz). We denote the ball centroid in frame $k$ by
$\mathbf{p}_k = (x_k, y_k)^\top$ in pixel coordinates, with the image origin at
the top-left corner and the $y$-axis pointing downward.

### B. Ball Detection

Each frame $I_k$ is passed to a YOLOv8 single-class detector
$\mathcal{D}_\text{ball}$ trained on cricket ball imagery. The raw detection set
is filtered by confidence, bounding-box area, and aspect ratio to suppress
spurious activations on the seam, fielders' clothing, and pitch markings:

$$
\mathcal{B}_k = \Big\{ b_i \;\Big|\; c_i \ge \tau_c,\;
A_\text{min} \le a_i \le A_\text{max},\;
r_\text{min} \le \rho_i \le r_\text{max} \Big\},
$$

where for the $i$-th candidate box $b_i = (x_1, y_1, x_2, y_2)$, the confidence
is $c_i$, the area $a_i = (x_2 - x_1)(y_2 - y_1)$, and the aspect ratio
$\rho_i = (x_2 - x_1)/(y_2 - y_1)$. We use $\tau_c = 0.07$,
$A_\text{min} = 10$, $A_\text{max} = 4000$ px$^2$, and
$r_\text{min} = 0.4$, $r_\text{max} = 2.0$. The detected centroid is
$\mathbf{p}_k = \big(\tfrac{x_1+x_2}{2}, \tfrac{y_1+y_2}{2}\big)$.

The confidence threshold $\tau_c$ is set deliberately low because broadcast
motion blur depresses ball confidence; the geometric area/ratio gates recover
precision lost to the low $\tau_c$.

### C. Ball Tracking via Kalman Filtering

To maintain a continuous ball state under the sparse and intermittent detections
characteristic of broadcast footage, we employ a linear Kalman filter with a
constant-velocity motion model. The state vector augments position with velocity,

$$
\mathbf{s}_k = \big[ x_k,\; y_k,\; \dot{x}_k,\; \dot{y}_k \big]^\top,
$$

and evolves according to

$$
\mathbf{s}_k = \mathbf{F}\,\mathbf{s}_{k-1} + \mathbf{w}_k,
\qquad
\mathbf{z}_k = \mathbf{H}\,\mathbf{s}_k + \mathbf{v}_k,
$$

with process noise $\mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$ and
measurement noise $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$. The
state-transition and measurement matrices, for time step $\Delta t = 1/f$, are

$$
\mathbf{F} =
\begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix},
\qquad
\mathbf{H} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}.
$$

The filter is initialised with $\mathbf{P}_0 = 500\,\mathbf{I}_4$ and uses
$\mathbf{Q} = 0.1\,\mathbf{I}_4$, $\mathbf{R} = 10\,\mathbf{I}_2$. The standard
predict–update recursion is applied:

**Predict:**
$$
\hat{\mathbf{s}}_k = \mathbf{F}\,\mathbf{s}_{k-1},
\qquad
\hat{\mathbf{P}}_k = \mathbf{F}\,\mathbf{P}_{k-1}\,\mathbf{F}^\top + \mathbf{Q}.
$$

**Update** (when a measurement $\mathbf{z}_k$ is available):
$$
\mathbf{y}_k = \mathbf{z}_k - \mathbf{H}\,\hat{\mathbf{s}}_k,
\qquad
\mathbf{S}_k = \mathbf{H}\,\hat{\mathbf{P}}_k\,\mathbf{H}^\top + \mathbf{R},
$$

$$
\mathbf{K}_k = \hat{\mathbf{P}}_k\,\mathbf{H}^\top\,\mathbf{S}_k^{-1},
\qquad
\mathbf{s}_k = \hat{\mathbf{s}}_k + \mathbf{K}_k\,\mathbf{y}_k,
$$

$$
\mathbf{P}_k = (\mathbf{I}_4 - \mathbf{K}_k\,\mathbf{H})\,\hat{\mathbf{P}}_k .
$$

In parallel, a windowed finite-difference estimator provides a robust
short-horizon velocity used by the contact-detection stage. Over a sliding
window of the $N$ most recent associated positions ($N = 6$),

$$
\hat{v}_x = \frac{(x_N - x_1)\,f}{N-1},
\qquad
\hat{v}_y = \frac{(y_N - y_1)\,f}{N-1},
$$

and the instantaneous pixel speed is $s_k = \sqrt{\hat{v}_x^2 + \hat{v}_y^2}$.

A *phantom-speed* guard rejects track corruption from distant false detections:
if $s_k > 800$ px·frame$^{-1}$ the track is reinitialised, since such speeds are
physically impossible for a ball within the field of view.

### D. Data Association

When multiple ball candidates survive detection filtering, the candidate nearest
to the Kalman-predicted position $\hat{\mathbf{p}}_k = \mathbf{H}\hat{\mathbf{s}}_k$
is associated to the track, subject to a gating radius $\tau_d$:

$$
i^\star = \underset{i \,:\, \lVert \mathbf{c}_i - \hat{\mathbf{p}}_k \rVert_2 < \tau_d}{\arg\min}
\;\big\lVert \mathbf{c}_i - \hat{\mathbf{p}}_k \big\rVert_2 ,
$$

where $\mathbf{c}_i$ is the $i$-th candidate centroid and $\tau_d = 150$ px. If
no candidate lies within the gate, the frame is treated as a miss and the track
coasts on the prediction.

### E. Interpolation of Short Detection Gaps

Because the empirical detection hit-rate is low (Section IV-B), short gaps are
bridged by linear interpolation between the last associated position
$\mathbf{p}_\text{last}$ and the current Kalman prediction
$\tilde{\mathbf{p}}_k$. For a gap of length $g$ frames with $g \le G$ ($G = 5$),
the synthesised measurement is

$$
\hat{\mathbf{p}}_k = (1 - \alpha)\,\mathbf{p}_\text{last} + \alpha\,\tilde{\mathbf{p}}_k,
\qquad
\alpha = \frac{g}{G + 1}.
$$

This recovers a usable post-contact trajectory in the presence of motion blur
without hallucinating long ballistic arcs.

### F. Batsman Pose Estimation

The on-strike batsman is localised by a YOLOv8-pose network
$\mathcal{D}_\text{pose}$, which returns, for each detected person $j$, a
bounding box $\mathbf{B}_j$, a detection confidence, and 17 COCO keypoints
$\{\mathbf{k}_{j,m}\}_{m=1}^{17}$ with per-keypoint confidences. The left and
right wrists correspond to COCO indices $m = 9$ and $m = 10$.

Robust batsman selection is critical, since the bowler and umpire are frequently
larger or more central than the batsman. We therefore select the batsman using
two reliable cues and explicitly avoid any "largest person" heuristic:

1. **Bat-box containment.** Let $\mathbf{B}^\text{bat}$ be the highest-confidence
   bat detection with confidence $\ge \tau_\text{bat}^{\text{pose}} = 0.55$ whose
   centre lies in the lower portion of the frame
   ($y > 0.40\,H$). The selected batsman is the person whose box contains the
   bat-box centre $\mathbf{m}^\text{bat}$:
   $$
   j^\star = \underset{j \,:\, \mathbf{m}^\text{bat} \in \mathbf{B}_j}{\arg\max}\; \text{conf}(j).
   $$

2. **IoU continuity.** When no qualifying bat box is present, the batsman from
   the previous frame is propagated by maximising the intersection-over-union
   with the prior box $\mathbf{B}^\star_{k-1}$:
   $$
   j^\star = \arg\max_j\; \text{IoU}(\mathbf{B}_j, \mathbf{B}^\star_{k-1}),
   \quad \text{subject to } \text{IoU} \ge \tau_\text{IoU} = 0.20 .
   $$

If neither cue fires, no batsman is reported for that frame; consequently the
wrist-velocity trigger is silent during the bowler's run-up, which is the
desired behaviour.

### G. Bat–Ball Contact Detection

Contact is signalled by the disjunction of three parallel triggers, evaluated in
priority order and subject to a refractory period (cooldown) of
$\tau_\text{cd} = 90$ frames between accepted contacts:

$$
\mathcal{C}_k = \big( T_1 \vee T_2 \vee T_3 \big)
\;\wedge\; \big(k - k_\text{last} > \tau_\text{cd}\big).
$$

**Trigger 1 — Bat-box proximity (highest precision).** A contact is declared if
the ball centroid lies within a margin $m = 35$ px of the bat bounding box
$\mathbf{B}^\text{bat} = (x_1, y_1, x_2, y_2)$:

$$
T_1 :\quad
(x_1 - m \le x_k \le x_2 + m)\;\wedge\;(y_1 - m \le y_k \le y_2 + m).
$$

**Trigger 2 — Trajectory curvature and acceleration spike.** A sharp change in
ball heading coincident with an abrupt speed change indicates a deflection. Over
a half-window $W = 5$, define the recent and prior displacement vectors
$\mathbf{u} = \mathbf{p}_k - \mathbf{p}_{k-W}$ and
$\mathbf{v} = \mathbf{p}_{k-W} - \mathbf{p}_{k-2W}$. The curvature angle is

$$
\kappa_k = \arccos\!\left(
\frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert\,\lVert \mathbf{v} \rVert}
\right),
$$

and the acceleration magnitude over a window $w = 3$ is

$$
a_k = \big| \bar{s}_{[k-w+1,\,k]} - \bar{s}_{[k-2w+1,\,k-w]} \big|,
$$

where $\bar{s}_{[\cdot]}$ is the mean pixel speed over the indexed window. The
trigger fires when

$$
T_2 :\quad \kappa_k > 10^\circ \;\wedge\; a_k > 1.5 \;\wedge\; s_k > 5 .
$$

**Trigger 3 — Wrist-velocity spike.** A bat swing produces a characteristic
spike in wrist speed. Let $\mathbf{w}^{(t)}_L$ and $\mathbf{w}^{(t)}_R$ be the
left/right wrist positions over a history of $\Lambda = 5$ frames. The peak
wrist speed is

$$
s^w_k = \max_{j \in \{L,R\}}\;\max_{t}\;
\big\lVert \mathbf{w}^{(t)}_j - \mathbf{w}^{(t-1)}_j \big\rVert_2 .
$$

To prevent firing on practice swings and guard-taking, the trigger is gated on a
recent ball observation within $\Delta_b = 15$ frames:

$$
T_3 :\quad s^w_k \ge \tau_w \;\wedge\; (k - k_\text{ball}) \le \Delta_b,
\qquad \tau_w = 80\ \text{px·frame}^{-1}.
$$

The three triggers are complementary: $T_1$ is precise but requires a clean bat
detection; $T_2$ requires a tracked post-contact arc; $T_3$ fires from body
kinematics alone and therefore recovers contacts that the appearance-based
triggers miss.

### H. Post-Contact Trajectory Reconstruction

On a contact at frame $k_c$, post-contact ball positions are accumulated over an
analysis window of $\Delta_w = 45$ frames. Duplicate and stale positions are
removed (a point is retained only if it differs from its predecessor by more than
2 px in either axis), yielding a sequence $\{\mathbf{q}_i\}_{i=1}^{M}$. The
sequence is smoothed with a length-5 moving-average kernel,

$$
\tilde{\mathbf{q}}_i = \frac{1}{5} \sum_{\ell=0}^{4} \mathbf{q}_{i+\ell},
$$

and a degree-2 polynomial is fit by ordinary least squares to model the
quasi-parabolic ground projection,

$$
\hat{y}(x) = c_2 x^2 + c_1 x + c_0,
\qquad
\mathbf{c} = \arg\min_{\mathbf{c}} \sum_{i} \big( y_i - (c_2 x_i^2 + c_1 x_i + c_0) \big)^2 ,
$$

from which a forward extrapolation of 40 samples gives the predicted future
trajectory (clipped to the frame bounds).

### I. Wagon-Wheel Shot Direction

The shot direction is derived from the net post-contact displacement
$\Delta x = x_M - x_1$ and $\Delta y = y_1 - y_M$ (the sign of $\Delta y$
inverts the downward image axis so that "up the ground" is positive). The raw
heading is

$$
\theta_\text{raw} = \operatorname{atan2}(\Delta y,\, \Delta x) \in [0^\circ, 360^\circ),
$$

which is then rotated into the standard wagon-wheel frame, in which
$0^\circ$ points straight back toward the bowler and angles increase clockwise:

$$
\theta = (90^\circ - \theta_\text{raw}) \bmod 360^\circ .
$$

### J. Hybrid Shot Classification

We classify the stroke by fusing a geometric rule base with a deep
spatiotemporal classifier. The geometric classifier $g(\theta)$ maps the
wagon-wheel angle to one of ten canonical strokes via the piecewise table

$$
g(\theta) =
\begin{cases}
\text{Straight Drive}, & \theta \in [345^\circ, 360^\circ)\cup[0^\circ,15^\circ)\\
\text{Off Drive}, & \theta \in [15^\circ, 40^\circ)\\
\text{Cover Drive}, & \theta \in [40^\circ, 75^\circ)\\
\text{Square Cut}, & \theta \in [75^\circ, 105^\circ)\\
\text{Late Cut}, & \theta \in [105^\circ, 150^\circ)\\
\text{Edge}, & \theta \in [150^\circ, 200^\circ)\\
\text{Leg Glance}, & \theta \in [200^\circ, 250^\circ)\\
\text{Pull Shot}, & \theta \in [250^\circ, 285^\circ)\\
\text{Flick}, & \theta \in [285^\circ, 320^\circ)\\
\text{On Drive}, & \theta \in [320^\circ, 345^\circ).
\end{cases}
$$

The deep classifier $\mathcal{F}_\text{CNN}$ is an EfficientNet-B0 backbone with
a GRU temporal head, consuming 30 frames resized to $224\times224$ and returning
a label $S_\text{CNN}$ with softmax confidence $p_\text{CNN}$. Because the deep
model occasionally produces high-confidence but kinematically impossible labels
on follow-through frames, the deep prediction overrides geometry only when it is
both highly confident *and* directionally consistent with the observed ball
motion. Defining the circular angular distance

$$
\Delta\theta(a, b) = \min\big(|a - b|,\; 360^\circ - |a - b|\big),
$$

and the per-class reference angle $\phi(S)$ (the centre of each stroke's sector),
the fused decision is

$$
\hat{S} =
\begin{cases}
S_\text{CNN}, & p_\text{CNN} \ge 0.97 \;\wedge\; \Delta\theta\big(\phi(S_\text{CNN}), \theta\big) \le 100^\circ,\\[4pt]
g(\theta), & \text{otherwise.}
\end{cases}
$$

This makes geometry the primary classifier — grounded in the measured ball
trajectory — while permitting the deep model to refine the label only when it is
near-certain and physically plausible. Events lacking a reconstructed trajectory
are discarded, because direction-free deep predictions were found to false-fire
on the static batting stance between deliveries (Section IV-E).

### K. Stroke Distance Estimation

A first-order metric stroke distance is obtained by summing the smoothed
post-contact segment lengths under a fixed pixel-to-metre scale
$\lambda = 18.5/520 \approx 0.0356$ m·px$^{-1}$ (derived from the 20.12 m pitch
length spanning $\approx 520$ px in the broadcast geometry):

$$
D = \lambda \sum_{i=2}^{M} \big\lVert \tilde{\mathbf{q}}_i - \tilde{\mathbf{q}}_{i-1} \big\rVert_2 .
$$

The ball release/delivery speed is reported with perspective correction
$\sigma(y)$ and exponential smoothing,

$$
v_k = s_k\,\lambda\,\sigma(y_k)\cdot 3.6, \qquad
\bar{v}_k = \beta\,v_k + (1-\beta)\,\bar{v}_{k-1}, \quad \beta = 0.25,
$$

with $v$ in km·h$^{-1}$.

### L. Delivery Segmentation and Event Generation

A new delivery is hypothesised when the ball reappears in the upper region of
the frame ($y < 0.35\,H$, the bowler's release zone) after an absence of at
least 30 frames:

$$
\text{NewDelivery}_k :\quad
(\text{missed}_k \ge 30)\;\wedge\;(y_k < 0.35\,H).
$$

On a new delivery, the delivery index $d$ is incremented and all per-stroke state
(track, overlays, wrist history) is reset, ensuring that each ball receives an
independent identifier. For every accepted contact the system emits a JSON
record

$$
e = \big\langle\, d,\; k_c,\; t_c,\; \theta,\; \hat{S},\; p_\text{CNN},\; D,\;
\mathbf{q}_1,\; \{\hat{y}(x)\} \,\big\rangle,
$$

comprising the delivery index, contact frame and timestamp
$t_c = k_c/f$, wagon-wheel angle, fused shot label, classifier confidence,
estimated distance, contact point, and predicted forward trajectory.

---

## IV. RESULTS AND DISCUSSION

### A. Experimental Setup

Experiments were conducted on a single workstation equipped with an NVIDIA
GeForce RTX 5060 Laptop GPU (Blackwell architecture, compute capability 12.0,
CUDA 13.2) running Ubuntu. The pipeline executes within a Python 3.11 / PyTorch
2.12 environment; the deep shot classifier runs under ONNX Runtime with the CUDA
execution provider. The ball, bat, and pose detectors are YOLOv8 variants; the
shot classifier is an EfficientNet-B0 + GRU network exported to ONNX.

The evaluation clip is a 4 min 10 s broadcast segment (timestamps 19:50–24:00 of
a full-match recording) at $1920\times1080$ and 29.97 Hz. As the source is
AV1-encoded, it was transcoded to H.264 prior to processing. The segment
contains **six bowler deliveries with corresponding bat–ball contacts**, which
serve as ground truth for contact-level evaluation.

### B. Ball Detection Performance

Profiling the ball detector over the 7 492-frame clip yields a per-frame
detection hit-rate of **7.9 %** (592 frames with at least one valid ball
detection). Among detected balls, confidence averages $0.45$ (range
$0.10$–$0.86$). The distribution of consecutive-miss runs is highly skewed: the
median gap is 8 frames while the mean is 39.5 frames, with the longest gap
spanning 832 frames (a passage with no ball in view). Crucially, **36 % of gaps
are $\le 3$ frames and 46 % are $\le 5$ frames**, which directly motivates the
$G = 5$ interpolation horizon of Section III-E — a single, cheap interpolation
rule recovers nearly half of all gaps without extrapolating into long blind
intervals.

This sparsity is the dominant error source in the pipeline and is attributable
to broadcast motion blur, small ball size, and frequent occlusion. It is the
principal reason a *multi-trigger* contact-detection design is necessary:
appearance-only triggers cannot be relied upon when the ball is invisible for
seconds at a time.

### C. Contact Detection and the Necessity of Multiple Triggers

Across the clip, the three triggers collectively fired on every one of the six
ground-truth contacts. The wrist-velocity trigger $T_3$ proved decisive: on
numerous deliveries the ball was undetected at the instant of contact, so
triggers $T_1$ and $T_2$ — both of which require ball evidence — were silent,
and only the body-kinematic trigger fired. The recorded peak wrist speeds at
true contacts ranged from $\sim 90$ to $\sim 280$ px·frame$^{-1}$, comfortably
above the $\tau_w = 80$ threshold, while the ball-recency gate ($\Delta_b = 15$)
suppressed practice swings during which no ball was present.

The batsman-selection logic (Section III-F) was validated qualitatively: prior
to introducing the bat-box-containment and IoU-continuity cues, the pose tracker
frequently locked onto the bowler during the run-up (the largest figure in
frame) and onto the square-leg umpire. Restricting anchoring to a
high-confidence, lower-frame bat box eliminated these failures, after which the
wrist keypoints remained on the batsman.

### D. Shot Classification and Hybrid Fusion

Requiring a reconstructed ball trajectory for every emitted event, the pipeline
produced **seven events for the six-ball clip** (one false positive, zero false
negatives). All seven carry a valid wagon-wheel angle and a high classifier
confidence; Table I summarises the output.

**TABLE I. Emitted shot events on the six-ball evaluation clip.**

| Delivery $d$ | $t_c$ (s) | Wagon angle $\theta$ | Fused label $\hat{S}$ | $p_\text{CNN}$ |
|:---:|:---:|:---:|:---|:---:|
| 0 | 4.5 | 265.0° | Flick | 100 % |
| 0 | 9.8 | 166.3° | Edge | 98 % |
| 1 | 18.0 | 200.1° | Leg Glance | 100 % |
| 2 | 31.3 | 285.0° | Pull Shot | 86 % |
| 2 | 34.9 | 94.3° | Lofted Drive | 100 % |
| 5 | 161.0 | 148.4° | Late Cut | 97 % |
| 7 | 215.4 | 157.2° | Edge | 100 % |

The value of the directional-consistency gate in the fusion rule (Section III-J)
is illustrated by several intercepted overrides. In one delivery the deep model
predicted "Straight Drive" with $p_\text{CNN} = 99.8\%$, yet the measured ball
direction was $\theta = 156.1^\circ$ (behind square), giving an angular
discrepancy of $156^\circ \gg 100^\circ$; the override was rejected and the
geometric label "Edge" retained. Similarly a "Lofted Drive" prediction at
$\theta = 226.9^\circ$ (leg side) was overruled. Without this gate, the deep
model's high-confidence-but-implausible predictions on follow-through frames
would have corrupted the labels.

### E. Ablation: Event-Admission Policy

We compared three event-admission policies against the six-ball ground truth, as
summarised in Table II. Admitting deep-classifier predictions even in the
absence of a ball trajectory ("CNN-only fallback") inflated the count to 20
events, the majority being false positives in which the network classified the
static batting stance between deliveries as a stroke. Requiring a reconstructed
trajectory for every event reduced this to seven (one false positive).
Additionally enforcing a single event per delivery yielded five events — but at
the cost of a *false negative*, because the imperfect delivery segmentation
occasionally merges two physical balls into one delivery index (notably at the
clip start, where all pre-first-delivery contacts share $d = 0$).

**TABLE II. Effect of event-admission policy (ground truth: 6 contacts).**

| Policy | Events | FP | FN |
|:---|:---:|:---:|:---:|
| CNN-only fallback (no trajectory required) | 20 | 14 | 0 |
| Trajectory required (adopted) | 7 | 1 | 0 |
| Trajectory + one-per-delivery | 5 | 0 | 1 |

We adopt the trajectory-required policy. For an analytics use case, an over-count
with a visible, inspectable false positive is preferable to an under-count that
silently discards a genuine stroke; a false positive is recoverable by downstream
review whereas a false negative is lost information. The single residual false
positive arises not from the classifier but from the delivery-segmentation stage,
and is therefore best addressed by improving segmentation rather than by
tightening the event gate.

### F. Computational Performance

The full pipeline — four neural networks (ball, bat, pose, shot) plus tracking
and analysis — sustains an end-to-end throughput of approximately
**16.6 frames·s$^{-1}$** at $1920\times1080$ on the target GPU. Memory footprint
was bounded by storing the 60-frame classifier buffer at the network input
resolution ($224\times224$) rather than full resolution, which reduced the
rolling buffer from $\sim$360 MB to $\sim$18 MB and eliminated out-of-memory
failures on multi-minute clips.

### G. Limitations and Future Work

Three limitations frame the future work. **(i) Ball-detection sparsity** (7.9 %
hit-rate) remains the bottleneck: it both starves the trajectory-based triggers
and limits the fraction of events with a measurable wagon-wheel angle.
Retraining the ball detector on broadcast-specific, motion-blurred data, or
fusing a high-frame-rate appearance cue, is the highest-impact next step.
**(ii) Delivery segmentation** relies on a single spatial heuristic (ball
reappearing in the upper frame) and merges balls when the release zone is not
cleanly observed; a learned, audio- or scoreboard-aware segmenter would remove
the residual false positive. **(iii) Ground-truth scale** — the present
validation is on a single six-ball clip; a larger annotated corpus is required
to report quantitative precision/recall and per-class shot-classification
accuracy with confidence intervals. These directions notwithstanding, the
results demonstrate that a multi-trigger design, anchored by body-kinematic
contact detection and a physically-gated hybrid classifier, reliably recovers
bat–ball contacts under the severe detection sparsity of real broadcast footage.

---

*Notation summary.* $\mathbf{p}_k$: ball centroid; $\mathbf{s}_k$: Kalman state;
$s_k$: pixel speed; $\kappa_k$: trajectory curvature; $a_k$: acceleration
magnitude; $s^w_k$: peak wrist speed; $\theta$: wagon-wheel angle;
$g(\theta)$: geometric classifier; $S_\text{CNN}, p_\text{CNN}$: deep label and
confidence; $\hat{S}$: fused label; $D$: stroke distance; $d$: delivery index;
$\lambda$: pixel-to-metre scale; $f$: frame rate.
