"""YOLOv3/COCO vehicle detection, postprocessing, and NMS."""
import cv2
import numpy as np

from src.components.models import Detection, VEHICLE_LABELS
from src.components.video_network_setup import DEFAULT_MODEL_DIR, load_network


class YoloVehicleDetector:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, confidence=0.5, nms_threshold=0.4):
        if not 0 < confidence <= 1 or not 0 < nms_threshold < 1:
            raise ValueError("Confidence and NMS thresholds must be in (0, 1]")
        self.net, self.output_layers, self.labels = load_network(model_dir)
        self.confidence = confidence
        self.nms_threshold = nms_threshold

    def detect(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        boxes, scores, classes = [], [], []
        for output in outputs:
            for candidate in output:
                probabilities = candidate[5:]
                class_id = int(np.argmax(probabilities))
                if class_id not in VEHICLE_LABELS:
                    continue
                score = float(candidate[4] * probabilities[class_id])
                if score < self.confidence:
                    continue
                cx, cy, bw, bh = candidate[:4] * np.array([width, height, width, height])
                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(width, int(cx + bw / 2))
                y2 = min(height, int(cy + bh / 2))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(score)
                classes.append(class_id)
        if not boxes:
            return []
        # NMS per class to avoid suppressing overlapping cars and trucks.
        selected = []
        for class_id in VEHICLE_LABELS:
            indices = [idx for idx, cls in enumerate(classes) if cls == class_id]
            if not indices:
                continue
            keep = cv2.dnn.NMSBoxes(
                [boxes[i] for i in indices], [scores[i] for i in indices],
                self.confidence, self.nms_threshold,
            )
            for chosen in np.asarray(keep).reshape(-1):
                i = indices[int(chosen)]
                selected.append(Detection(tuple(boxes[i]), class_id, scores[i]))
        return selected
