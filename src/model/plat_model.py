import cv2
from src.utils.base_detector import BaseDetector
import os
from src import conf


class PlatDetector(BaseDetector):
    def __init__(self):
        super().__init__(os.path.join(conf.BASE_URL, "weights", conf.PLAT_MODEL_NAME), conf.PLAT_CLASSES)


# testing getting plat image
if __name__ == "__main__":
    path_model = os.path.join(conf.BASE_URL, "weights", "license_plate_detector.pt")
    plat_model = PlatDetector()
    img = cv2.imread(r"C:\Users\User\PycharmProjects\OCR smart-parking\exp\assets\168.jpg")
    results = plat_model.predict(img)
    cv2.imshow("Plat", plat_model.get_image(img, results))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
