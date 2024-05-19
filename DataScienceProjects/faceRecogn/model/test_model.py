import cv2
import joblib
import numpy as np
import pywt
from matplotlib import pyplot as plt
model = joblib.load('face_model.pkl')

# celebs = ["", "lionel_messi", "maria_sharapova", "roger feeder",
#   "serena_williams", "virat_kohli"]
celebs = {"lionel_messi": 1, "maria_sharapova": 2,
          "roger_federer": 3, "serena_williams": 4, "virat_kohli": 5}
print(type(celebs))


def w2d(img, mode='haar', level=1):
    imArray = img
    # datatype conversion to grayscale
    imArray = cv2.cvtColor(imArray, code=cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute co-efficients
    coeff = pywt.wavedec2(imArray, mode, level=level)

    # process
    coeffs_H = list(coeff)
    coeffs_H[0] *= 0

    # reconstruct
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


X = []
# filename = './dataset/cropped/lionel_messi/lionel_messi1.png'
# filename = './test_images/Virat_test_model1.jpg'
# filename = './test_images/maria_test_model3.jpg'
filename = './test_images/sharapova1.jpg'
img = cv2.imread(filename)
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gr, cmap='gray')
face_cascade = cv2.CascadeClassifier(
    './opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    './opencv/haarcascades/haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gr, 1.2, 5)
print(faces)
for (x, y, w, h) in faces:
    X = []

    rol_color = img[y:y+h, x:x+w]
    # face_rect = cv2.rectangle(img, (x, y), (x+w, y+w), (5, 250, 0), 2)
    plt.imshow(rol_color)
    scalled_raw_img = cv2.resize(rol_color, (32, 32))
    img_har = w2d(scalled_raw_img, 'db1')
    scalled_img_har = cv2.resize(img_har, (32, 32))
    plt.imshow(scalled_raw_img)
    plt.show()
    combined_img = np.vstack((scalled_raw_img.reshape(
        32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
    X.append(combined_img)
    X = np.array(X).reshape(len(X), 4096).astype(float)
    celeb = model.predict(X)
    print(celeb)
    if (len(celeb) > 0):
        print(list(celebs.keys())[list(celebs.values()).index(celeb[0])])
    # plt.imshow(face_rect, cmap='gray')
    # plt.show()
# img = cv2.imread(filename)
# scalled_raw_img = cv2.resize(img, (32, 32))
# img_har = w2d(scalled_raw_img, 'db1')
# scalled_img_har = cv2.resize(img_har, (32, 32))
# # # vertically put the vevelet image
# combined_img = np.vstack((scalled_raw_img.reshape(
#     32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
# X.append(combined_img)
# X = np.array(X).reshape(len(X), 4096).astype(float)

# print(model.predict(X))
