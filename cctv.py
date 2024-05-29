import cv2
import numpy as np
# import cvui
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
path = 'rtsp://admin:admin12345@192.168.0.105:554/cam/realmonitor?channel=1&subtype=0'

cv = cv2.VideoCapture()
cv.open(path)
cv.set(3, 640)  # ID number for width is 3
cv.set(4, 480)  # ID number for height is 480
cv.set(10, 100)  # ID number for brightness is 10qq
# cvui.init('screen')

out = cv2.VideoWriter('filename.avi',
                      cv2.VideoWriter_fourcc(*'MJPG'),
                      10, (640, 480))
while True:
    success, current_cam = cv.read()
    if success == True:
        out.write(current_cam)
        dim = (640, 480)
        Full_frame = cv2.resize(
            current_cam, dim, interpolation=cv2.INTER_AREA)
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)

        frame = cv2.cvtColor(Full_frame, cv2.COLOR_BGR2GRAY)
        # ret, frame = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(Full_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow('screen', Full_frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

out.release()
cv.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
