from src.controller import TextRecognition, CarDetector, PlatDetector


class OCRController:
    def __init__(self):
        self.text_recognition = TextRecognition()
        self.car_detector = CarDetector()
        self.plat_detector = PlatDetector()

    def get_car(self, frame):
        return self.car_detector.get_image(frame, self.car_detector.predict(frame))

    def get_plat(self, frame):
        return self.plat_detector.get_image(frame, self.plat_detector.predict(frame))

    def get_text(self, frame):
        return self.text_recognition.text_read(frame)

    def run(self, frame):
        car_image = self.get_car(frame)
        if car_image.size == 0:
            return "No car detected"

        plat_image = self.get_plat(car_image)
        if plat_image.size == 0:
            return "No plat detected"
        # TODO: to heavy to run easy OCR per frame
        # text = self.get_text(plat_image)
        return ""
