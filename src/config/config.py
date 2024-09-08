from pathlib import Path

class CONFIG:
    BASE_URL = Path(__file__).parent.parent.resolve()
    CAR_CLASSES = [2, 5, 7]
    PLAT_CLASSES = []
    # CAR_MODEL_NAME = "yolov8n.pt"
    CAR_MODEL_NAME = "yolov8n.onnx"
    PLAT_MODEL_NAME = "license_plate_detector.onnx"
    # PLAT_MODEL_NAME = "license_plate_detector.pt"


conf = CONFIG()
