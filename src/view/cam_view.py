import cv2

def show_cam(text , frame):
    cv2.imshow(text, frame)

def show_box(frame, x1:int,x2:int,y1:int,y2:int, color:tuple):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

