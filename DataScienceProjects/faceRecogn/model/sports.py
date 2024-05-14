import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

cv2.destroyAllWindows()
# filename = './test_images/3faces.jpg'
filename = './test_images/sharapova2.jpg'  # no eyes
img = cv2.imread(filename)
cv2.imshow('sharapova', img)
cv2.waitKey()
# print(img.shape)
# (555, 700, 3) w h RGB=3
# plt.imshow(img)
# plt.show()

# convert to gray = 2 color
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


plt.imshow(gr, cmap='gray')
plt.show()
# print(gr.shape)
face_cascade = cv2.CascadeClassifier(
    './opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    './opencv/haarcascades/haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gr, 1.2, 5)
print(faces)
# [[357  39 231 231]]  1 array means 1 face X,Y, WIDTH, HEIGHT face location
# from 3 faces [[384  87 181 181] [698 102 165 165] [ 84 113 195 195]] 3 array of face location
for (x, y, w, h) in faces:
    face_rect = cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0), 2)
    rol_gary = gr[y:y+h, x:x+w]
    rol_color = face_rect[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(rol_gary)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(rol_color, (ex, ey), (ex+ew, ey+eh), (25, 255, 00), 2)


plt.imshow(face_rect)
plt.show()
