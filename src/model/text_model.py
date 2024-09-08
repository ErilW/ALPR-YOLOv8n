import cv2
from easyocr import Reader
from src import conf
from src.utils.image_handler import correct_skew
import numpy as np
from src.model.realsgan_model import GanModel


class TextRecognition:
    def __init__(self):
        self.reader = Reader(['en'], gpu=False)
        self.super_res = GanModel()

    def _pre_processing_image(self, plat_image) -> np.ndarray:
        # rotate image
        rotated_image = correct_skew(plat_image)
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 13, 15, 15)
        img = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return img

    def text_read(self, frame) -> str:
        img = self._pre_processing_image(frame)
        texts = self.reader.readtext(img, text_threshold=0.5)
        text = []
        bounding_box = []
        for _text in texts:
            text.append(_text[1])


        return "".join(text)

    def get_text_location(self, frame) -> list:
        img = self._pre_processing_image(frame)
        texts = self.reader.readtext(img, slope_ths=0.0)
        bounding = []
        for _text in texts:
            if any(char in _text[1] for char in ['0', 'O', 'Q', '8', 'D', 'o', 'q', 'd']):
                bounding.append(_text[0])

        print(bounding)
        return bounding,  img


if __name__ == "__main__":
    Tr = TextRecognition()
    i = cv2.imread(str(conf.BASE_URL) + r"\assets\sample_text.png")
    print(Tr.text_read(i))

