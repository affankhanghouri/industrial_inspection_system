"""Small CPU runtimes for statically exported Ultralytics detection models."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class ExportDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def _names_from_value(value: object) -> dict[int, str]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, list):
        value = dict(enumerate(value))
    if not isinstance(value, dict):
        raise RuntimeError("exported YOLO model does not contain class names")
    return {int(class_id): str(name) for class_id, name in value.items()}


class ExportedYoloRuntime:
    """Run one static ONNX or OpenVINO YOLO export without PyTorch inference."""

    def __init__(
        self,
        backend: str,
        model_path: Path,
        *,
        image_size: int,
        cpu_threads: int,
        confidence: float,
        requested_class_names: tuple[str, ...],
        nms_iou: float = 0.70,
    ) -> None:
        if backend not in {"onnxruntime", "openvino"}:
            raise ValueError(f"unsupported exported YOLO backend: {backend!r}")
        self.backend = backend
        self.model_path = model_path
        self.image_size = image_size
        self.confidence = confidence
        self.nms_iou = nms_iou
        self._session: Any
        self._input: Any
        self._output: Any

        if backend == "onnxruntime":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._input = self._session.get_inputs()[0].name
            self._output = self._session.get_outputs()[0].name
            names = _names_from_value(
                self._session.get_modelmeta().custom_metadata_map.get("names")
            )
            input_shape = self._session.get_inputs()[0].shape
        else:
            from openvino import Core

            xml_path = self._openvino_xml(model_path)
            core = Core()
            model = core.read_model(xml_path)
            self._session = core.compile_model(
                model,
                "CPU",
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": cpu_threads,
                    "NUM_STREAMS": "1",
                },
            )
            self._input = self._session.input(0)
            self._output = self._session.output(0)
            input_shape = list(self._input.shape)
            metadata_path = xml_path.parent / "metadata.yaml"
            if not metadata_path.is_file():
                raise RuntimeError(f"OpenVINO metadata is missing: {metadata_path}")
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            names = _names_from_value(metadata.get("names"))

        if list(input_shape[-2:]) != [image_size, image_size]:
            raise ValueError(
                f"export input is {list(input_shape[-2:])}, configured image size is "
                f"{image_size}"
            )
        requested = {name.strip().lower() for name in requested_class_names}
        self.class_names = names
        self.class_ids = frozenset(
            class_id
            for class_id, name in names.items()
            if name.strip().lower() in requested
        )
        if not self.class_ids:
            raise RuntimeError(
                f"export contains none of the requested classes: {sorted(requested)}"
            )

    @staticmethod
    def _openvino_xml(path: Path) -> Path:
        if path.is_file() and path.suffix.lower() == ".xml":
            return path
        if path.is_dir():
            candidates = sorted(path.glob("*.xml"))
            if len(candidates) == 1:
                return candidates[0]
        raise FileNotFoundError(f"could not resolve one OpenVINO XML model from {path}")

    def warmup(self) -> None:
        self._infer(np.zeros((1, 3, self.image_size, self.image_size), np.float32))

    def detect(self, image: np.ndarray) -> tuple[ExportDetection, ...]:
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        output = self._infer(tensor)
        return self._postprocess(
            output,
            original_width=image.shape[1],
            original_height=image.shape[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        if self.backend == "onnxruntime":
            return np.asarray(self._session.run([self._output], {self._input: tensor})[0])
        return np.asarray(self._session([tensor])[self._output])

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        height, width = image.shape[:2]
        scale = min(self.image_size / width, self.image_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        pad_x = (self.image_size - resized_width) / 2.0
        pad_y = (self.image_size - resized_height) / 2.0
        left = round(pad_x - 0.1)
        right = round(pad_x + 0.1)
        top = round(pad_y - 0.1)
        bottom = round(pad_y + 0.1)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(
            padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        )
        tensor /= 255.0
        return tensor, scale, float(left), float(top)

    def _postprocess(
        self,
        output: np.ndarray,
        *,
        original_width: int,
        original_height: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[ExportDetection, ...]:
        predictions = np.squeeze(output, axis=0)
        if predictions.ndim != 2:
            raise RuntimeError(f"unexpected YOLO output shape: {output.shape}")
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        requested_ids = sorted(self.class_ids)
        class_scores = predictions[:, 4:]
        candidate_scores = class_scores[:, requested_ids]
        best_requested = np.argmax(candidate_scores, axis=1)
        confidences = candidate_scores[
            np.arange(candidate_scores.shape[0]), best_requested
        ]
        keep = confidences >= self.confidence
        if not np.any(keep):
            return ()
        selected = predictions[keep, :4]
        confidences = confidences[keep]
        class_ids = np.asarray(requested_ids, dtype=np.int64)[best_requested[keep]]

        cx, cy, width, height = selected.T
        x1 = (cx - width / 2.0 - pad_x) / scale
        y1 = (cy - height / 2.0 - pad_y) / scale
        x2 = (cx + width / 2.0 - pad_x) / scale
        y2 = (cy + height / 2.0 - pad_y) / scale
        x1 = np.clip(x1, 0, original_width)
        y1 = np.clip(y1, 0, original_height)
        x2 = np.clip(x2, 0, original_width)
        y2 = np.clip(y2, 0, original_height)

        kept_indices: list[int] = []
        for class_id in requested_ids:
            indices = np.flatnonzero(class_ids == class_id)
            if not len(indices):
                continue
            boxes = [
                [
                    float(x1[index]),
                    float(y1[index]),
                    float(x2[index] - x1[index]),
                    float(y2[index] - y1[index]),
                ]
                for index in indices
            ]
            scores = [float(confidences[index]) for index in indices]
            selected_indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.confidence, self.nms_iou
            )
            kept_indices.extend(int(indices[int(index)]) for index in selected_indices)

        kept_indices.sort(key=lambda index: (-float(confidences[index]), index))
        return tuple(
            ExportDetection(
                class_id=int(class_ids[index]),
                class_name=self.class_names[int(class_ids[index])],
                confidence=float(confidences[index]),
                x1=float(x1[index]),
                y1=float(y1[index]),
                x2=float(x2[index]),
                y2=float(y2[index]),
            )
            for index in kept_indices
        )
