# Milestone 2 benchmark results

Hardware: Intel Core i5-6300U, 2 physical cores / 4 logical threads, 16 GB RAM, CPU only. Input video is 1280×720 at 30 FPS per camera. Detector input is 640 px with confidence 0.40 and person-only filtering.

Measurements use `/usr/bin/time -v`. CPU percentages are aggregate process CPU, where 100% is one logical core. Peak RSS includes Python, OpenCV, PyTorch, the model, and live frame storage.

## Video-only baseline

Command:

```bash
/usr/bin/time -v .venv/bin/python scripts/run_cameras.py --duration 5 --stats-interval 2
```

- Existing tests: 9 passed
- Camera FPS: 30.0–30.1
- Latest-frame age: 5–35 ms
- CPU: 115%
- Peak RSS: 161,936 KB

## Detector-only

Each run used three warm-up calls followed by 18 timed calls over representative midpoint frames from the actual videos.

| Torch threads | Mean | Median | p95 | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 109.63 ms | 108.03 ms | 131.03 ms | 9.12 FPS |
| 2 | 110.21 ms | 108.48 ms | 136.06 ms | 9.07 FPS |
| 4 | 101.49 ms | 99.76 ms | 119.00 ms | 9.85 FPS |

Four threads provide only an 8% isolated gain and occupy all logical CPUs. It was not selected for the live system.

## One live camera

Command:

```bash
/usr/bin/time -v .venv/bin/python scripts/run_detection.py --threads 1 --cameras CAM-001 --duration 10 --stats-interval 2
```

- Camera FPS: 30.0
- AI FPS: 7.08
- Mean inference: 140.0 ms
- Mean end-to-end frame age: 157.5 ms
- Last end-to-end age: 194.2 ms
- Inference calls: 72
- Skipped frames: 232
- Errors: 0
- CPU: 163%
- Peak RSS: 504,196 KB

## Three live cameras, selected one-thread configuration

Command:

```bash
/usr/bin/time -v .venv/bin/python scripts/run_detection.py --threads 1 --duration 15 --stats-interval 3
```

| Camera | Camera FPS | AI FPS | Calls | Mean inference | Mean end-to-end age | Last age | Skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAM-001 | 30.0 | 1.99 | 30 | 171.5 ms | 190.2 ms | 207.7 ms | 420 |
| CAM-002 | 29.9 | 1.93 | 29 | 172.4 ms | 191.4 ms | 164.0 ms | 419 |
| CAM-003 | 30.0 | 1.93 | 29 | 165.0 ms | 183.5 ms | 189.1 ms | 419 |

- Global AI throughput: 5.85 FPS
- Global mean inference: 169.7 ms
- Inference errors: 0
- CPU: 183%
- Peak RSS: 576,756 KB (about 563 MB)
- All clips looped and all components shut down cleanly

A second 15-second run with two Torch threads produced 5.78 global FPS, 182% CPU, and 573,360 KB peak RSS. Because it did not improve long-run throughput, the default remains one thread.

## Visual sanity check

`scripts/generate_detection_samples.py` scanned 12 positions in each source and retained the candidate with the most confident people. Visual inspection confirmed boxes around people in every selected image:

- CAM-001 frame 80: 4 people
- CAM-002 frame 135: 3 people
- CAM-003 frame 81: 3 people

Images are in `debug/detections/`. No source video was modified.
