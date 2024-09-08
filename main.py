import argparse
import cv2
from src.controller.ocr_controller import OCRController

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', "--image", type=str, required=True, help="Path to image")
    arg = vars(args.parse_args())

    image_path = arg['image']
    controller = OCRController()

    cap = cv2.VideoCapture(image_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        text = controller.run(frame)
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
