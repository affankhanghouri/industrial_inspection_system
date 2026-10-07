# Milestone 4 — Helmet Observation Results

## Scope

Milestone 4 adds helmet detection and associates helmet observations with the
confirmed person tracks produced by Milestone 3. It does not add temporal PPE
state, compliance decisions, alerts, violations, zones, persistence, or a user
interface.

The output is deliberately observational:

- `OBSERVED` means a helmet detection was associated with a confirmed person
  track in the same camera frame.
- `UNOBSERVED` means no helmet detection was associated with that confirmed
  track in that processed frame.
- `UNOBSERVED` is not equivalent to "not wearing a helmet" and is not a safety
  violation.
- Tentative and lost tracks are omitted from PPE association output.

PPE inference is enabled for `CAM-001` only. All cameras continue to run person
detection and tracking.

## Model provenance

| Field | Value |
| --- | --- |
| Checkpoint | `models/yolov8n_ppe_construction.pt` |
| Upstream repository | `baskarmother/yolov8-ppe-construction` |
| Upstream file | `best.pt` |
| Pinned revision | `3213ed51de90cbc76e577e6944e84f7c74343526` |
| File size | 6,258,474 bytes |
| SHA-256 | `8714b4b2bbde95b3a07dcdbe873995e34742b5ce628464a4da232721d4691ffe` |
| Declared checkpoint license | MIT |
| Base architecture | YOLOv8n |
| Input size | 640 |
| Relevant embedded class | index 4, `hardhat` |
| Runtime | Ultralytics 8.4.172, PyTorch 2.14.1, CPU |

The checkpoint contains both positive and negative-looking classes. Milestone 4
loads and exposes only class 4 (`hardhat`). In particular, `no-hardhat` is not
used to infer absence or non-compliance. Full provenance, dataset notes, and the
separate Ultralytics runtime licensing caveat are recorded in
`models/PPE_MODEL.md`.

## Detector strategy evaluation

Two CPU strategies were evaluated against the supplied looping CAM-001 clip.

| Strategy | Setup | Mean latency | Approx. throughput | Result |
| --- | --- | ---: | ---: | --- |
| Full-frame | One PPE inference on the complete 640-pixel frame | 130.73 ms | 7.65 FPS | Selected |
| Per-person head crops | Expanded upper 40% of each person box; one 320-pixel inference per crop | 357.64 ms | 2.80 FPS | Rejected |

The crop strategy multiplied inference calls by the person count, was about
2.7 times slower in the measured sequence, missed valid helmets, and produced
duplicate detections. Full-frame inference produced three stable helmet
detections on each of the 15 initially sampled frames and allows the association
stage to handle track ownership separately.

## Association method

For each confirmed track with a current-frame source detection:

1. Build a head region from the upper 30% of the person box.
2. Expand it by 20% horizontally and 5% vertically.
3. Require the helmet center to fall within that region.
4. Require at least 50% of the helmet area to overlap the region.
5. Compute a score using 70% containment and 30% normalized center proximity.
6. Require a minimum score of 0.55.
7. Reject observations whose two best track scores are within 0.10.
8. Greedily apply a one-to-one helmet/track assignment.

Association is allowed only when camera ID and frame ID match exactly. The PPE
coordinator reads the one retained person frame used for person inference; it
does not associate results to a newly captured or merely nearby frame.

## Bounded scheduling

All cross-thread state is latest-only:

- latest tracking result: one slot;
- latest PPE inference result: one slot;
- latest PPE association result: one slot;
- optional processed display image: one slot;
- statistics: fixed-size deques.

There is no unbounded frame or result queue. PPE inference is rate-limited to
one eligible frame per second by default. When the detector is busy, intermediate
tracking frames are overwritten and the next inference uses the newest exact
retained frame.

The one-second cap was adopted after an uncapped three-camera run showed CPU
saturation and reduced person throughput. With the cap, PPE still samples at
1 FPS while preserving the existing multi-camera pipeline.

## Automated validation

The final suite contains 56 passing tests. Milestone 4 coverage includes:

- PPE configuration defaults and validation;
- backend-neutral detector filtering and coordinate conversion;
- model class-name validation;
- helmet/head geometry and edge cases;
- confirmed, tentative, and lost track behavior;
- camera/frame mismatch rejection;
- one-to-one matching and ambiguity rejection;
- explicit `OBSERVED`/`UNOBSERVED` semantics;
- preservation of raw unassociated detections;
- latest-only result stores;
- coordinator camera enablement, exact-frame behavior, failure isolation, and
  rate limiting;
- clean thread shutdown.

Command:

```bash
pytest -q
```

## Isolated detector benchmark

The PPE model was warmed up and measured over 30 CPU iterations at image size
640 and confidence 0.25.

| Metric | Result |
| --- | ---: |
| Mean latency | 108.61 ms |
| Median latency | 106.15 ms |
| p95 latency | 121.35 ms |
| Throughput | 9.21 FPS |

Command:

```bash
.venv/bin/python scripts/benchmark_ppe_detector.py \
  --warmup 3 \
  --iterations 30
```

## Pre-change baselines

Short contemporary baselines were captured before integrating the PPE thread.
They complement the longer canonical Milestone 3 measurements.

| Run | Camera ingest | Person FPS | Person latency | Tracking FPS | Tracking latency | CPU | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAM-001, 5 s | 30.1 | 6.76 | 144.9 ms | 6.56 | 0.336 ms | 149% | 495 MiB |
| Three cameras, 6 s | 29.7–30.1 each | 6.11 global | 160.6 ms | 5.94 global | 0.258 ms | 166% | 560 MiB |

## Uncapped integration measurements

These runs intentionally removed the PPE interval to expose contention.

| Run | Person FPS | PPE FPS | Person latency | PPE latency | CPU | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAM-001, 8 s | 4.79 | 4.67 | 206.3 ms | 193.8 ms | 198% | 539 MiB |
| Three cameras, 8 s | 5.46 global | 1.86 | 180.9 ms | 227.6 ms | 199% | 618.5 MiB |

The one-camera person rate fell about 29% from its short pre-change baseline.
This was treated as resource starvation, not as an acceptable default.

A three-camera trial with the one-second interval recovered global person
throughput to 6.15 FPS while maintaining PPE at 0.99 FPS, motivating the final
default.

## Final CAM-001 validation

Default configuration, 12-second run:

| Metric | Result |
| --- | ---: |
| Camera ingest | 30.0 FPS |
| Person detection | 7.25 FPS |
| Tracking | 7.17 FPS |
| PPE detection | 1.00 FPS |
| Person mean latency | 136.9 ms |
| PPE latency, mean / median / p95 | 189.5 / 183.4 / 242.6 ms |
| Tracking mean latency | 0.312 ms |
| Association mean / median / p95 | 0.137 / 0.148 / 0.175 ms |
| Latest PPE result age | 385.0 ms |
| CPU | 187% |
| Peak RSS | 544 MiB |
| Frame mismatches / processing errors | 0 / 0 |

The video looped once and processing continued normally.

## Final three-camera validation

Default configuration, 20-second run. PPE remained enabled only for CAM-001.

| Metric | CAM-001 | CAM-002 | CAM-003 | Global |
| --- | ---: | ---: | ---: | ---: |
| Camera ingest FPS | 30.0 | 30.0 | 30.0 | — |
| Person detection FPS | 2.10 | 2.10 | 2.05 | 6.25 |
| Tracking FPS | 2.10 | 2.05 | 2.05 | 6.20 |
| Mean person latency | 161.4 ms | 153.8 ms | 162.3 ms | 159.2 ms |
| PPE FPS | 1.00 | disabled | disabled | 1.00 |

Additional global results:

| Metric | Result |
| --- | ---: |
| PPE latency, mean / median / p95 | 232.5 / 220.9 / 301.5 ms |
| Tracking mean latency | 0.262 ms |
| Association mean / median / p95 | 0.137 / 0.148 / 0.211 ms |
| Latest PPE result age | 604.2 ms |
| CPU | 219% |
| Peak RSS | 622 MiB |
| Frame mismatches / processing errors | 0 / 0 |

All three videos looped during the run and recovered normally. The process
exited with status zero.

## Comparison with canonical Milestone 3

| Three-camera metric | Milestone 3 | Milestone 4 | Change |
| --- | ---: | ---: | ---: |
| Global person FPS | 6.72 | 6.25 | -7.0% |
| Per-camera person FPS | 2.19–2.24 | 2.05–2.10 | about -4% to -8.5% |
| Mean person latency | 148.1 ms | 159.2 ms | +7.5% |
| Mean tracking latency | 0.256 ms | 0.262 ms | +0.006 ms |
| CPU | 206% | 219% | +13 percentage points |
| Peak RSS | 568 MiB | 622 MiB | +54 MiB |

The final global person rate also slightly exceeded the short contemporary
pre-change baseline (6.25 versus 6.11 FPS). The second loaded model costs about
54 MiB of resident memory. The configured rate cap prevents the severe
throughput loss seen in the uncapped test.

## Manual sequence validation

The CAM-001 debug sequence is in `debug/ppe/`. It contains five annotated frames,
a contact sheet, and `helmet_history.txt`.

Observed sequence behavior:

- three upper workers consistently receive helmet observations once their
  tracks become confirmed;
- the lower worker track remains `UNOBSERVED` in the sampled frames;
- no helmet is assigned to that lower track;
- raw detections are preserved as unassociated while tracks are tentative;
- all three raw detections associate after confirmation;
- a transient duplicate tentative person track is omitted from PPE output.

These results describe detector observations only and do not establish PPE
compliance.

## Failure handling and shutdown

Per-frame detector or association failures are counted and logged without
terminating camera, person-detection, tracking, or PPE worker threads. Exact
frame mismatches are rejected and counted. A SIGINT integration run exited with
status zero after orderly cleanup.

## Reproduction commands

```bash
# One enabled camera
.venv/bin/python scripts/run_ppe.py --cameras CAM-001 --duration 12

# All configured cameras; PPE still runs only on CAM-001
.venv/bin/python scripts/run_ppe.py --duration 20

# Regenerate visual evidence
.venv/bin/python scripts/generate_ppe_debug.py
```

## Remaining limitations

- Helmet observations are frame-local and can flicker with detector confidence,
  occlusion, or tracking-box jitter.
- `UNOBSERVED` is intrinsically ambiguous and must not be interpreted as
  confirmed absence.
- Only one short prerecorded scene was available for qualitative validation.
- CPU inference is resource-constrained; the cap prioritizes the existing
  person/tracking pipeline over PPE sampling frequency.
- The model and Ultralytics runtime have separate license declarations; a
  deployment owner must complete the applicable license review.

A future milestone can add a bounded, explicitly defined temporal observation
accumulator. It should remain separate from policy, compliance, and alert logic.
