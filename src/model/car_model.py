import cv2
from src.utils.base_detector import BaseDetector
import os
from src import conf


class CarDetector(BaseDetector):
    def __init__(self):
        super().__init__(os.path.join(conf.BASE_URL, "weights", "yolov8n.pt"), conf.CAR_CLASSES)


if __name__ == "__main__":
    car_model = CarDetector()
    img = cv2.imread(r"C:\Users\User\PycharmProjects\OCR smart-parking\exp\assets\168.jpg")
    results = car_model.predict(img)
    cv2.imshow("Plat", car_model.get_image(img, results))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


