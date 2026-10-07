# Milestone 4 PPE checkpoint

Local file: `models/yolov8n_ppe_construction.pt`

- Origin: `baskarmother/yolov8-ppe-construction` on Hugging Face
- Pinned repository revision: `3213ed51de90cbc76e577e6944e84f7c74343526`
- Original filename: `best.pt`
- Download URL: `https://huggingface.co/baskarmother/yolov8-ppe-construction`
- SHA-256: `8714b4b2bbde95b3a07dcdbe873995e34742b5ce628464a4da232721d4691ffe`
- File size: 6,258,474 bytes
- Checkpoint license declared by the model card: MIT
- Architecture declared by the model card: YOLOv8n detection
- Training dataset declared by the model card:
  `keremberke/construction-safety-object-detection`
- Training details declared by the model card: 640×640, 100 epochs, SGD
- Runtime validated here: Ultralytics 8.4.172 with PyTorch 2.14.1 CPU

The embedded checkpoint metadata exposes 17 classes. Class 4 is `hardhat`;
class 6 is `no-hardhat`. Milestone 4 requests only class 4 from the backend and
maps it to the platform-owned `HELMET` observation type. It deliberately does
not expose or interpret the checkpoint's `no-hardhat` class.

The MIT declaration applies to the checkpoint as published by its author. The
Ultralytics software/runtime used to execute it has separate AGPL-3.0 and
commercial licensing options. Product deployment requires an independent
license review; this file is provenance documentation, not legal advice.
