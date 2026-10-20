"""Load legacy Darknet YOLOv3 weights/config into OpenCV DNN."""
from pathlib import Path
import cv2

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "Data"


def load_network(model_dir: Path = DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    weights = model_dir / "yolov3.weights"
    cfg = model_dir / "yolov3.cfg"
    names = model_dir / "coco.names"
    missing = [p.name for p in (weights, cfg, names) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing YOLOv3 assets: {', '.join(missing)} in {model_dir}. "
            "Run: python -m src.download_models"
        )
    labels = names.read_text(encoding="utf-8").splitlines()
    if len(labels) <= 7:
        raise ValueError("coco.names is missing expected COCO class labels")
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    layers = net.getUnconnectedOutLayersNames()
    return net, layers, labels
