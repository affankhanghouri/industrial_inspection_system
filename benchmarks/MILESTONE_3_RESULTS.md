# Milestone 3 Tracking Results

Date: 2026-10-04  
Target: Intel i5-6300U, CPU-only Ubuntu environment  
Model: `models/yolo26n.pt`, Ultralytics 8.4.172, 640 px, person-only, one
PyTorch thread and one OpenCV thread

## Scope and design decision

Milestone 3 adds anonymous camera-local multi-person trajectories after the
validated Milestone 2 person detector. It does not add PPE, zones, rules,
violations, evidence, backend services, persistent identity, cross-camera
matching, or ReID.

`TimeAwareIoUPersonTracker` was selected because the runtime receives sparse
structured detections at roughly 2.2 observations/second/camera rather than
every video frame. The tracker:

- owns state for exactly one camera;
- predicts boxes with smoothed coordinate velocity;
- associates by predicted IoU and normalized center distance;
- uses monotonic elapsed observation time for prediction and expiration;
- tolerates non-consecutive global frame IDs;
- keeps short misses as `LOST` and expires stale tracks;
- resets on a source-loop change, frame-ID regression, or source-time rollback;
- uses no appearance model, private Ultralytics tracker API, or new heavy
  dependency.

One `TrackingCoordinator` consumes each camera's single latest inference slot.
It publishes to one single latest tracking slot per camera. There is no tracking
work queue; `stored_result_count` is at most one. A tracking error resets only
the affected camera tracker and does not terminate video or inference.

## Validated configuration

```yaml
tracker:
  backend: time_aware_iou
  minimum_detection_confidence: 0.40
  minimum_iou: 0.05
  maximum_center_distance: 1.50
  maximum_lost_seconds: 1.75
  minimum_confirmed_hits: 2
  maximum_prediction_seconds: 1.00
  velocity_smoothing: 0.60
runtime:
  idle_wait_ms: 5
```

The initial 1.25-second lost window fragmented two CAM-002 trajectories after
1.5-second gaps (track 2 became 4; track 3 became 5). Increasing only the lost
window to 1.75 seconds preserved both original IDs. CAM-002 then used exactly
three IDs for its three visible people. This was the only evidence-driven
tracker tuning.

## Commands

Full tests:

```bash
.venv/bin/python -m pytest -q
```

Real-video continuity artifacts:

```bash
.venv/bin/python scripts/generate_tracking_debug.py
```

One-camera live benchmark spanning the 7.4-second CAM-001 loop:

```bash
/usr/bin/time -v .venv/bin/python scripts/run_tracking.py \
  --cameras CAM-001 --duration 12 --stats-interval 4
```

Three-camera live benchmark spanning all source loops:

```bash
/usr/bin/time -v .venv/bin/python scripts/run_tracking.py \
  --duration 20 --stats-interval 5
```

Graceful signal check:

```bash
timeout --preserve-status --signal=INT 8 \
  .venv/bin/python scripts/run_tracking.py \
  --cameras CAM-001 --stats-interval 2
```

The signal check returned status 0 after orderly component shutdown.

## Test result

```text
36 passed in 2.73s
```

Coverage includes all Milestone 1/2 tests plus tracker interface and records,
camera-local namespaces, sparse frame gaps, two-person separation, short-miss
recovery, elapsed-time expiry, no-detection input, loop/restart discontinuity
reset, isolated camera reset/error, latest-only overwrite, and clean close.

## One-camera live result

Run interval excludes the preceding model warm-up in application FPS/latency
counters. `/usr/bin/time` CPU/RSS covers the complete process including warm-up.

| Metric | CAM-001 / global |
|---|---:|
| Camera ingestion | 30.1 FPS |
| Detector throughput | 6.99 FPS |
| Final tracking update rate | 6.91 FPS |
| Steady tracking update rate at 8 s | 7.37 FPS |
| Mean inference latency | 141.9 ms |
| Mean tracking latency | 0.325 ms |
| Median tracking latency | 0.326 ms |
| p95 tracking latency | 0.486 ms |
| Maximum tracking latency | 0.698 ms |
| Mean observation interval | 0.142 s |
| Maximum source frame-ID gap | 8 |
| Final active / confirmed / lost | 4 / 4 / 0 |
| Tracks created, cumulative across reset | 9 |
| Expired tracks | 0 |
| Tracker resets | 1 |
| Tracking errors | 0 |
| Aggregate CPU | 167% |
| Peak RSS | 502,760 KiB (~491 MiB) |

At shutdown, the inference scheduler can complete one last result before the
tracking coordinator is stopped, which accounts for the small 6.99 versus 6.91
final-rate difference. During steady snapshots the rates matched.

## Three-camera live result

| Metric | CAM-001 | CAM-002 | CAM-003 | Global |
|---|---:|---:|---:|---:|
| Camera ingestion FPS | 29.9 | 30.0 | 29.9 | — |
| Detector FPS | 2.24 | 2.24 | 2.24 | 6.72 |
| Final tracking updates/s | 2.19 | 2.24 | 2.24 | 6.67 |
| Mean inference ms | 146.5 | 148.5 | 149.2 | 148.1 |
| Mean tracking ms | 0.329 | 0.199 | 0.241 | 0.256 |
| Median tracking ms | — | — | — | 0.235 |
| p95 tracking ms | 0.483 | 0.270 | 0.351 | 0.419 |
| Maximum tracking ms | — | — | — | 1.166 |
| Mean observation interval s | 0.442 | 0.444 | 0.442 | — |
| Maximum frame-ID gap | 19 | 18 | 18 | — |
| Final active / confirmed / lost | 4 / 4 / 0 | 1 / 1 / 2 | 3 / 3 / 0 | — |
| Tracks created | 13 | 6 | 6 | 25 |
| Expired tracks | 0 | 0 | 0 | 0 |
| Resets | 2 | 1 | 1 | 4 |
| Tracking errors | 0 | 0 | 0 | 0 |

Complete-process resource results:

- aggregate CPU: 206%;
- peak RSS: 581,648 KiB (about 568 MiB);
- steady 15-second snapshot: detector and tracking both 6.66 FPS;
- clean exit status: 0.

`expired=0` in these looping-video runs is expected: people leave near clip end
and the source discontinuity resets the tracker before the 1.75-second expiry
window elapses. Elapsed dead-track expiry is independently covered by a unit
test.

## Milestone 2 comparison

| Three-camera metric | Milestone 2 | Milestone 3 | Change |
|---|---:|---:|---:|
| Camera FPS/camera | 29.9–30.0 | 29.9–30.0 | unchanged |
| Global detector FPS | 5.85 | 6.72 | +14.9% |
| AI/tracking FPS/camera | 1.93–1.99 | 2.19–2.24 | higher |
| Mean inference | 165.0–172.4 ms | 148.1 ms global | lower in this run |
| Aggregate CPU | 183% | 206% | +23 percentage points |
| Peak RSS | ~563 MiB | ~568 MiB | ~+5 MiB |

The runs are not a controlled CPU-isolation experiment, so throughput and CPU
variation should not be attributed solely to tracking. The tracking operation
was 0.256 ms mean versus 148.1 ms inference (about 0.17%) and memory stayed
within about 1% of the Milestone 2 peak. There is no material tracking overhead
or ingestion/detector regression.

The one-camera comparison is similarly close: Milestone 2 measured 30.0 camera
FPS, 7.08 AI FPS, 140.0 ms inference, 163% CPU, and ~492 MiB RSS; Milestone 3
measured 30.1, 6.99, 141.9 ms, 167%, and ~491 MiB respectively.

## Manual continuity review

Artifacts are in `debug/tracking/`; the authoritative observation listing is
`debug/tracking/track_history.txt`. Sampling used the actual detector on every
15th source frame (2 FPS), not hand-authored boxes.

### CAM-001

- Upper-left worker: track 1 at all 15 observations, 0.0–7.0 s.
- Upper-center worker: track 2 at all 15 observations, 0.0–7.0 s.
- Upper-right worker: track 3 at all 15 observations, 0.0–7.0 s.
- Lower moving worker: track 4 from first detection at 2.0 s through 6.0 s
  (nine hits), then `LOST` at 6.5 and 7.0 s after leaving view.
- At 6.0 s, YOLO returned a second overlapping box on the same partially
  occluded lower worker. It created tentative track 5 for one observation. This
  is a visible false duplicate, not a switch away from confirmed track 4; track
  5 never confirmed.

### CAM-002

- Initially left/then moving upper worker: track 1 through detections from
  0.0–8.5 s; it was `LOST` at 9.0/9.5 s when no matching detection existed.
- Initially right upper worker: track 2 for the full sequence. It survived
  detector misses at 2.0/2.5 s and reacquired at 3.0 s, missed at 6.0 s and
  reacquired at 6.5 s, and missed at 7.5 s then reacquired at 8.0 s.
- Lower worker entering from the left: track 3 from 2.0 s onward. It survived
  misses at 7.5/8.0 s and reacquired as track 3 at 8.5 s.
- No visible ID switch remained after tuning. The pre-tuning fragment IDs 4 and
  5 no longer occur.

### CAM-003

- Foreground/moving worker: track 1 at all 20 observations, 0.0–9.5 s.
- Left worker: track 2 at all 20 observations, 0.0–9.5 s.
- Center worker moving left: track 3 at all 20 observations, 0.0–9.5 s.
- No visible switch, fragmentation, or duplicate was observed.

No numeric MOTA, HOTA, IDF1, or ID-switch metric is reported because tracking
ground truth is unavailable.

## Loop reset and isolation validation

Offline history generation explicitly replayed source frame 0 with loop count
1. All three processors reported `source loop changed 0->1`, reset, and emitted
fresh local IDs from 1.

The live three-camera run observed:

```text
[CAM-001] tracker reset: source loop changed 0->1
[CAM-002] tracker reset: source loop changed 0->1
[CAM-003] tracker reset: source loop changed 0->1
[CAM-001] tracker reset: source loop changed 1->2
```

CAM-001 reset at its 7.4-second boundary while CAM-002 and CAM-003 continued
their existing tracks until their own ~9.97-second boundaries. Reset counters
were CAM-001=2, CAM-002=1, CAM-003=1, proving that resets are local rather than
global. A second CAM-002 source loop occurred during shutdown after inference
had already stopped, so there was intentionally no tracking reset for an
unobserved transition.

## Remaining limitations

- Geometry-only association cannot resolve long occlusion, visually crowded
  crossings, abrupt motion, or unreliable detections as robustly as appearance
  tracking.
- A permissive lost window improves sparse-observation continuity but can leave
  a predicted `LOST` box briefly visible after a person exits.
- Detector duplicates can create tentative duplicate tracks, as seen once in
  CAM-001.
- IDs are reset-scoped and camera-local; they must never be presented as worker
  identity.
- Current validation is manual on three short clips without annotated tracking
  ground truth.

Recommended Milestone 4: add backend-neutral PPE observations and associate
them to confirmed camera-local person tracks, with separate quality and CPU
validation. Do not combine that milestone with zones, rules, or violations.
