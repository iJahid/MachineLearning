import datetime
import shutil
import os
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

cv2.destroyAllWindows()
face_cascade = cv2.CascadeClassifier(
    './opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    './opencv/haarcascades/haarcascade_eye.xml')


def get_cropped_2_eyes(filename):
    img = cv2.imread(filename)

    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gr, 1.3, 5)

    for (x, y, w, h) in faces:
        rol_gary = gr[y:y+h, x:x+w]
        rol_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(rol_gary)
        if (len(eyes) >= 2):
            return rol_color
        # for (ex, ey, ew, eh) in eyes:
        #    cv2.rectangle(rol_color, (ex, ey), (ex+ew, ey+eh), (25, 255, 00), 2)


# face_rect = get_cropped_2_eyes('./test_images/sharapova1.jpg')

# plt.imshow(face_rect)
# plt.show()
now = datetime.datetime.now()
print('------------------', now.time())
path_to_data = './dataset/'
path_to_cr_data = './dataset/cropped/'

img_dirs = []
for entry in os.scandir(path_to_data):
    if (entry.is_dir()):
        img_dirs.append(entry.path)

print(img_dirs)

if (os.path.exists(path_to_cr_data)):
    shutil.rmtree(path_to_cr_data)

os.mkdir(path_to_cr_data)

cropped_img_dir = []
celeb_file_names = {}
count = 1
for img_d in img_dirs:
    count = 1
    celeb_name = img_d.split('/')[-1]

    if 'cropped' in celeb_name:
        pass
    else:
        print(celeb_name)
        for entry in os.scandir(img_d):
            # print(entry.path)
            roi_color = get_cropped_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cr_data+celeb_name
                if not os.path.exists(cropped_folder):
                    os.mkdir(cropped_folder)
                    cropped_img_dir.append(cropped_folder)
                    print(cropped_folder)

                crop_file_name = celeb_name+str(count)+".png"
                crop_file_path = cropped_folder+"/"+crop_file_name
                print('cropping..', crop_file_path)
                cv2.imwrite(crop_file_path, roi_color)
                # celeb_file_names[celeb_name].append(crop_file_path)
                count += 1

print(celeb_file_names)
