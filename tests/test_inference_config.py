from pathlib import Path

from industrial_safety.inference import load_inference_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_inference_configuration_loads_and_resolves_model() -> None:
    config = load_inference_config(project_root=PROJECT_ROOT)

    assert config.detector.backend == "onnxruntime"
    assert config.detector.model_path == PROJECT_ROOT / "models" / "yolo26n_512.onnx"
    assert config.detector.model_path.is_file()
    assert config.detector.image_size == 512
    assert config.detector.cpu_threads == 2
    assert config.detector.device == "cpu"
    assert config.detector.classes == ("person",)
    assert config.runtime.scheduling == "round_robin"
