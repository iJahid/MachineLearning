import json
import joblib
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import pywt


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


# testing
filename = './dataset/cropped/lionel_messi/lionel_messi1.png'  # no eyes
img = cv2.imread(filename)

# im_har = w2d(img, mode='db1')
# plt.imshow(im_har, cmap='gray')
# plt.show()

# reshape for weveling fixed size 32 x 32
# scalled_raw_img = cv2.resize(img, (32, 32))
# img_har = w2d(scalled_raw_img, 'db1')
# scalled_img_har = cv2.resize(img_har, (32, 32))
# # # vertically put the vevelet image
# combined_img = np.vstack((scalled_raw_img.reshape(
#     32*32*3, 1), scalled_img_har.reshape(32*32, 1)))

# plt.imshow(scalled_raw_img)
# plt.show()
# --end testing

# now lets do wevelet on each crop (faces) images and make the dataset for model
X = []
Y = []
class_dict = {}
celeb_num = 1
for entry in os.scandir('./dataset/cropped'):
    celebname = entry.path.split('\\')[-1]
    print(celebname)
    class_dict[celebname] = celeb_num
    for cimg in os.scandir(entry.path):
        print('wevleting ..', cimg.path)
        img = cv2.imread(cimg.path)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(scalled_raw_img, 'db1')
        scalled_img_har = cv2.resize(img_har, (32, 32))
        # # vertically put the vevelet image
        combined_img = np.vstack((scalled_raw_img.reshape(
            32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
        X.append(combined_img)
        Y.append(celeb_num)
    celeb_num += 1

print(len(Y))
X = np.array(X).reshape(len(X), 4096).astype(float)
print(X.shape)
# 172
# (172, 4096)

# now X and Y are ready time to create model with scaling, standardscaler and choose model with classification reports and cross valid
# SVC,
# from sklearn.svm import  SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report

# split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10)

# scale
pipe = Pipeline([('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
# 0.813953488372093 which is 88%

# lets see clasification report
cl_r = classification_report(y_test, pipe.predict(X_test))
print(cl_r)
# we can see the messi's precision is 100% and for others its making mistake sometime to recognize
#  precision    recall  f1-score   support

#            1       1.00      0.71      0.83         7


# now lets try GridSearchCV for comparing other models
model_parms = {
    'svm': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1.10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']

        }
    },
    'Logistic_Regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {

            'logisticregression__C': [1, 5, 10]
        }
    },
    'lasso': {
        'model': Lasso(),
        'params': {
            'lasso__alpha': [1, 2],
            'lasso__selection': ['random', 'cyclic']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]

        }
    }
}
best_estimators = {}
scores = []
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for algo_name, config in model_parms.items():
    pipe = make_pipeline(StandardScaler(), config['model'])

    gs = GridSearchCV(pipe, config['params'],
                      cv=5, return_train_score=False)
    gs.fit(X, Y)
    scores.append({
        'model': algo_name,
        'best_score': gs.best_score_,
        'best_params': gs.best_params_
    })
    best_estimators[algo_name] = gs.best_estimator_

print(pd.DataFrame(scores, columns=['model', 'best_score', 'best_params']))
# print(best_estimators)
best_model = best_estimators['svm']
print(best_model.score(X_test, y_test))
# 88%

# lets see confusion metrix
conf_m = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(10, 7))
sn.heatmap(conf_m, annot=True)
plt.show()

# now save this model to physical path
joblib.dump(best_model, 'face_model.pkl')
with open('class_dict.json', 'w') as f:
    f.write(json.dumps(class_dict))
