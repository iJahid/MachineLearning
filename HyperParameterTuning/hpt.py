import math
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# plt.imshow(mpimg.imread('iris.png'))
# plt.show()
iris = load_iris()
# besic featurs
# ['DESCR', 'data', 'data_module', 'feature_names'=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
# 'filename', 'frame', 'target', 'target_names'=['setosa' 'versicolor' 'virginica']]
# print(iris.feature_names)


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df.flower.apply(lambda x: iris.target_names[x])


print(cross_val_score(SVC(kernel='linear', C=10,
      gamma='auto'), iris.data, iris.target, cv=5))
print(cross_val_score(SVC(kernel='rbf', C=10,
      gamma='auto'), iris.data, iris.target, cv=5))
print(cross_val_score(SVC(kernel='linear', C=20,
      gamma='auto'), iris.data, iris.target, cv=5))

kernel = ['rbf', 'linear']
C = [1, 5, 10, 20]
avg_score = {}

for kval in kernel:
    for cval in C:
        cv_score = cross_val_score(SVC(kernel=kval, C=cval,
                                       gamma='auto'), iris.data, iris.target, cv=5)
        avg_score[kval+'_' +
                  str(cval)] = round(np.average(cv_score) * 100, 2)

print(avg_score)

clf = GridSearchCV(SVC(gamma='auto'), {
                   'C': C, 'kernel': kernel}, cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
df = pd.DataFrame(clf.cv_results_)
print(df[['param_C', 'param_kernel', 'mean_test_score']])
print('best score', clf.best_score_, 'best param', clf.best_params_)
