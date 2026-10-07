# Industrial AI Safety Compliance Platform

A complete, local, CPU-only safety demonstration for three simulated CCTV
cameras. The application ingests looping MP4 sources, detects and tracks people,
evaluates helmet and polygon-entry rules, stores reviewable events and annotated
evidence, and serves a FastAPI dashboard.

For visual documentation of the complete design, see
[`docs/SYSTEM_DIAGRAMS.md`](docs/SYSTEM_DIAGRAMS.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_system.py
```

Large model files, CCTV clips, generated evidence, and the runtime database are
intentionally not committed. Before running the system, provide these local
assets at the paths used by the checked-in configuration:

```text
models/yolo26n_512.onnx
models/yolov8n_ppe_construction_512_openvino_model/
videos/camera_01_helmet_violation.mp4
videos/camera_02_loading_area.mp4
videos/camera_03_restricted_area.mp4
```

The PPE checkpoint source, pinned revision, checksum, and licensing caveats are
documented in [`models/PPE_MODEL.md`](models/PPE_MODEL.md). If different assets
are used, update `config/inference.yaml`, `config/ppe.yaml`, and
`config/cameras.yaml` accordingly. The application creates `data/` runtime
content automatically.

Open **http://127.0.0.1:8000**. Press Ctrl+C once to stop the API and every
pipeline worker cleanly. A bounded run can use `--duration 20`; `--no-api` is
available for headless diagnostics.

For a single preview containing only the three camera videos, run:

```bash
python scripts/run_system.py --display
```

CAM-001 and CAM-002 appear on the top row, with CAM-003 centered below them.
Each camera reads its own latest-frame buffer independently. The composite is
1280×720 and refreshes at a CPU-friendly preview rate while the inference
workers retain scheduling priority. For one-camera visual playback, run one of:

```bash
python scripts/run_system.py --display --camera CAM-001
python scripts/run_system.py --display --camera CAM-002
python scripts/run_system.py --display --camera CAM-003
```

Only the selected camera is decoded and processed. CAM-001 includes helmet/PPE
state; CAM-002 and CAM-003 include their configured polygons and zone state.
Confirmed events appear in the window. Press `q`, close the window, or press
Ctrl+C to shut down cleanly. The video advances from the camera's latest-frame
buffer while person/PPE/zone overlays update independently at their own rates;
overlay age is shown in the footer. `--camera` also works without `--display`.
For visual QA, `--display-screenshot path.png` saves the first rendered frame
containing a newly confirmed event.

## Final architecture

```text
3 camera workers -> one-slot latest frames -> shared ONNX person detector
  -> camera-local trackers
  -> CAM-001 OpenVINO helmet detector + helmet/track association
  -> CAM-002/CAM-003 polygon transition engines
  -> versioned rule engine + temporal confirmation/cooldown
  -> annotated evidence + SQLite events
  -> FastAPI -> local dashboard
```

The data path is latest-only: slow inference does not create an unbounded frame
queue. Person inference is deterministic round-robin, tracking IDs are anonymous
and local to one camera, and source-loop changes reset only that camera's tracker
and zone state.

## Demonstrated rules

| Camera | Rule | Confirmation | Event |
|---|---|---|---|
| CAM-001 | Helmet required | 2 `UNOBSERVED` samples in 3.5 s; an `OBSERVED` sample clears pending state | `HELMET_MISSING` |
| CAM-002 | Enter loading polygon | bottom-center transitions outside to inside for 2 samples | `LOADING_AREA_ENTRY` |
| CAM-003 | Enter restricted doorway | bottom-center transitions outside to inside for 2 samples | `RESTRICTED_AREA_ENTRY` |

Each rule has a 300-second cooldown. `UNOBSERVED` is weak single-frame evidence,
not by itself a no-helmet conclusion; only the temporal rule creates an event.
Zone rules fire on an entry transition, not continuously while a person remains
inside.

## Configuration

- `config/cameras.yaml` — IDs, names, enabled state, and video sources.
- `config/inference.yaml` — person ONNX Runtime model, 512 input, threshold,
  thread counts, and round-robin scheduling.
- `config/tracking.yaml` — camera-local matching, confirmation, and expiry.
- `config/ppe.yaml` — PPE OpenVINO model, helmet class mapping, association,
  CAM-001 enablement, and 1 FPS rate cap.
- `config/zones.yaml` — CAM-002 loading and CAM-003 doorway polygons plus entry
  and exit debounce.
- `config/rules.yaml` — rule/camera bindings, severity, temporal thresholds,
  versions, and cooldowns.

Selected person runtime: `models/yolo26n_512.onnx`, ONNX Runtime FP32, 512 px,
confidence 0.30, two CPU threads. Selected PPE runtime:
`models/yolov8n_ppe_construction_512_openvino_model`, OpenVINO FP32, 512 px,
confidence 0.25, two CPU threads. The selected inference path does not execute
PyTorch. Source checkpoints and licensing/provenance remain documented under
`models/`; deployment must review the applicable Ultralytics/model licenses.

## API and persistence

The default database is `data/safety.db`. Each confirmed event stores one
annotated JPEG and one short MP4 under
`data/evidence/YYYY-MM-DD/<event-id>/`. A bounded 10 FPS buffer retains up to
2.5 seconds before confirmation and captures up to 2.5 seconds afterward;
encoding runs on a bounded background writer. SQLite records both evidence
paths, survives application restarts, and supports `OPEN`, `RESOLVED`, and
`DISMISSED` review states.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | service health |
| GET | `/api/cameras` | camera and pipeline telemetry |
| GET | `/api/events` | recent events; optional `status` and `limit` |
| GET | `/api/events/{event_id}` | event detail |
| GET | `/api/events/{event_id}/evidence` | path-confined JPEG evidence |
| GET | `/api/events/{event_id}/clip` | path-confined MP4 evidence with range support |
| POST | `/api/events/{event_id}/resolve` | persist resolved review state |
| POST | `/api/events/{event_id}/dismiss` | persist dismissed review state |
| GET | `/api/stats` | event totals by status/type |

POST bodies accept `{"reason": "review note"}`. The dashboard shows three
plain camera-status indicators, four violation summaries, readable recent
events, screenshot/video evidence, basic status/camera filters, and review
controls without exposing inference or tracker diagnostics.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/generate_zone_debug.py
```

The final real-video acceptance run generated exactly one expected event per
camera before cooldown suppression. Generated evidence and dashboard captures
are intentionally excluded from version control. Full results and reproducible
distinctions between isolated model speed and application throughput are in
`FINAL_SYSTEM_REPORT.md` and `benchmarks/FINAL_PERFORMANCE_RESULTS.md`.

## Operating limits

This is a bounded demonstration on an Intel i5-6300U (two cores/four threads),
not a certified safety control. It uses looping local videos, not RTSP. Models
are generic and validated only on the supplied clips. There is no cross-camera
identity, facial recognition, RFID, behavior/fall analysis, vehicle analytics,
or cloud alerting. INT8 was not validated because its optional NNCF installation
failed when only about 85 MiB of disk remained; the environment rolled back and
the validated FP32 exports remain selected.
