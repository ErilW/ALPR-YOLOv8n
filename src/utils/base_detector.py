from ultralytics import YOLO
import numpy as np
import cv2


class BaseDetector:
    def __init__(self, path, classes: list, gpu=False):
        self.gpu = gpu
        self.model = YOLO(path) if gpu else YOLO(path)
        self.classes = classes

    def predict(self, *args):
        frame = self.preprocessing_image(args[0])
        conf = args[1] if len(args) > 1 else 0.5
        if self.gpu:
            results = self.model.predict(frame, verbose=False, conf=conf)

        else:
            results = self.model.predict(frame, verbose=False, conf=conf, device='cpu')
        return results

    def track(self, frame):
        results = self.model.track(frame, verbose=False)
        return results

    @staticmethod
    def get_image(frame, results) -> np.ndarray:
        boxes = results[0].boxes.xyxy.cpu().tolist()
        if not boxes:
            return np.array([])
        x1, y1, x2, y2 = map(int, boxes[0])
        return frame[y1:y2, x1:x2]

    @staticmethod
    def preprocessing_image(frame) -> np.ndarray:
        frame_process = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_process

    @staticmethod
    def get_centroid(results):
        boxes = results[0].boxes.xyxy.cpu().tolist()
        if not boxes:
            return None
        x1, y1, x2, y2 = map(int, boxes[0])
        return (x1 + x2) // 2, (y1 + y2) // 2
