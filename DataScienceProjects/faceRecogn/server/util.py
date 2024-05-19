import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
from matplotlib import pyplot as plt


def classify_image(image64data, filepath=None):
    imgs = get_crop_image_2_eyes(filepath, image64data)

    celebs = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(
            32*32*3, 1), scalled_har.reshape(32*32, 1)))
        len_img_array = 32*32*3+32*32
        final = combined_img.reshape(1, len_img_array).astype(float)

        celeb = __model.predict(final)[0]
        celebs.append({
            'celeb': celeb_name(celeb),
            'proba': np.round(__model.predict_proba(final)*100, 2).tolist()[0],
            'dict': __name_2_number

        })
    return celebs


def celeb_name(celeb_num):
    return __number_2_name[celeb_num]


def load_artifacts():
    print('loading artifacts..')
    global __model
    global __name_2_number
    global __number_2_name

    with open('./artifacts/class_dict.json', 'r') as f:
        __name_2_number = json.load(f)

        __number_2_name = {v: k for k, v in __name_2_number.items()}

    with open('./artifacts/face_model.pkl', 'rb') as f:
        __model = joblib.load(f)

    print('artifacts..loaded')


def get_image_from_base64(base64_text):
    enc_data = base64_text.split(',')[1]
    npar = np.frombuffer(base64.b64decode(enc_data), np.uint8)
    img = cv2.imdecode(npar, cv2.IMREAD_COLOR)
    return img


def get_crop_image_2_eyes(image_path, img64data):
    face_cascade = cv2.CascadeClassifier(
        './opencv/haarcascades/haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier(
        './opencv/haarcascades/haarcascade_eye.xml')

    if (image_path is not None):
        img = cv2.imread(image_path)
    else:
        img = get_image_from_base64(img64data)

    # plt.imshow(img)
    # plt.show()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if (len(faces) == 0):
    #     print('no face detected')
    # print(faces)
    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_org = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if (len(eyes) >= 2):
            # print(roi_org)
            cropped_faces.append(roi_org)

    return cropped_faces


def get_b64_text():
    # filename='messi64.txt'
    # filename = 'viratmessi64_2.txt'
    filename = 'viratmesi64.txt'
    with open(filename) as f:
        return f.read()


if __name__ == '__main__':
    pass
    # load_artifacts()
    # celebs = classify_image(get_b64_text(), filepath=None)
    # celebs = classify_image(None, filepath='Kohli-Messi.jpg')
    # celebs = classify_image(None, filepath='messi-and-kohli.png')

    # for c in celebs:
    #     print(c)
