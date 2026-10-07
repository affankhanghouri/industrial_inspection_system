# Final Performance Results

Measured on the target Intel i5-6300U CPU. Values are empirical for the supplied
clips and installed runtime; small run-to-run variation is expected.

## Person backend selection

| Backend | Input / threads | Mean ms | Median ms | p95 ms | FPS | Representative counts |
|---|---|---:|---:|---:|---:|---|
| PyTorch | 640 / 1 | 110.24 | 107.83 | 146.18 | 9.07 | 4/3/3 |
| PyTorch | 512 / default | 68.68 | 68.21 | 72.08 | 14.56 | 4/3/3 |
| ONNX Runtime | 640 / 2 | 115.20 | 110.19 | 169.89 | 8.68 | accepted counts |
| OpenVINO | 640 / 2 | 94.07 | 93.23 | 103.96 | 10.63 | accepted counts |
| ONNX Runtime | 512 / 1 | 93.91 | — | — | 10.65 | 4/3/3 |
| **ONNX Runtime** | **512 / 2** | **59.77** | **58.43** | **73.36** | **16.73** | **4/3/3** |
| ONNX Runtime | 512 / 4 | 81.14 | — | — | 12.32 | 4/3/3 |
| OpenVINO | 512 / 1 | 104.58 | — | — | 9.56 | 4/3/3 |
| OpenVINO | 512 / 2 | 63.04 | 62.46 | 69.30 | 15.86 | 4/3/3 |
| OpenVINO | 512 / 4 | 62.57 | 62.11 | 67.80 | 15.98 | 4/3/3 |

ONNX Runtime, 512 px, confidence 0.30, two threads is selected. It was the
fastest accepted variant and preserved the representative people. PyTorch 320
reached 26.85 FPS and 416 reached 14.49 FPS, but both detected only one CAM-002
worker and were rejected for correctness.

## PPE backend selection

| Backend | Input / threads | Mean ms | Median ms | p95 ms | FPS | Outcome |
|---|---|---:|---:|---:|---:|---|
| PyTorch | 512 | 77.59 | 80.09 | 95.73 | 12.89 | 45 observations |
| PyTorch | 640 | 108.61 | 106.15 | 121.35 | 9.21 | correct baseline |
| ONNX Runtime | 512 / 2 | 78.63 | 76.86 | 91.35 | 12.72 | 42 observations |
| **OpenVINO** | **512 / 2** | **77.87** | **76.29** | **101.09** | **12.84** | **42 observations; selected** |

OpenVINO provided validated exported inference with effectively equal speed to
the source checkpoint and avoids PyTorch at runtime. PyTorch 320 produced no
helmets; 416 returned incomplete observations, so neither was acceptable.

## Complete application

The API-enabled one-process run, including three video decoders, person and PPE
inference, tracking, rule/event processing, SQLite, FastAPI and dashboard work:

| Metric | Measured result |
|---|---:|
| Camera FPS, CAM-001/002/003 | 29.8 / 30.0 / 29.9 |
| Global person FPS | 10.11 |
| Approximate person FPS per camera | 3.24 |
| Mean person inference | 98.3 ms |
| PPE FPS | 1.00 |
| Mean PPE inference | 237.7 ms |
| Tracking ms, CAM-001/002/003 sample | 0.327 / 0.164 / 0.249 |
| Mean rule evaluation | 0.084 ms |
| Frame age ms, CAM-001/002/003 sample | 18.94 / 43.56 / 25.62 |
| CPU | 240% |
| Peak RSS | 435,248 KiB (~425 MiB) |

A second final run after the CAM-002 polygon acceptance correction produced the
same three expected scenarios at 9.83 global person FPS, 0.99 PPE FPS, 101.0 ms
person inference, 240.3 ms PPE inference, 0.737 ms rule evaluation, 271% CPU,
and 440,016 KiB (~430 MiB) peak RSS. Camera readers remained 29.9–30.0 FPS.

The final application result is approximately **10 global person FPS**, not
16.73 FPS. The latter is an isolated detector benchmark. The difference is
expected when all work contends for two physical cores.

## INT8 and disk status

No INT8 result exists. Installing optional NNCF failed with `ENOSPC` when only
about 85 MiB remained; pip rolled back and the working environment stayed
healthy. FP32 exported models are therefore the measured, validated selection.
At finalization the filesystem had approximately 1.5 GiB free and was 98% used.
Required videos, models, database, evidence, and reports were retained.
