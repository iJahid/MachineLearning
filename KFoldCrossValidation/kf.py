import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

to_predict = 10

digits = load_digits()
# plt.matshow(digits.images[to_predict])
# plt.show()
df = pd.DataFrame(digits.data)


X_tr, X_test, Y_tr, Y_test = train_test_split(
    df, digits.target, test_size=.3)


model = LogisticRegression(max_iter=1000)
model.fit(X_tr, Y_tr)
print('LogisticRegression accuracy : ', model.score(X_test, Y_test))

model = SVC()
model.fit(X_tr, Y_tr)
print('SVC accuracy : ', model.score(X_test, Y_test))


model = RandomForestClassifier(
    n_estimators=200)  # number of trees
model.fit(X_tr, Y_tr)
print('RandomForestClassifier accuracy : ', model.score(X_test, Y_test))


def get_score(model, x_train, y_train, x_test, t_test):
    model.fit(x_train, y_train)
    return model.score(x_test, t_test)


print('LogisticRegression accuracy : ',
      get_score(LogisticRegression(max_iter=1000), X_tr, Y_tr, X_test, Y_test))
print('RandomForestClassifier accuracy : ',
      get_score(RandomForestClassifier(), X_tr, Y_tr, X_test, Y_test))

print('SVC accuracy : ',
      get_score(SVC(), X_tr, Y_tr, X_test, Y_test))


# from train_test_split this scores changes every time it runs. its confusional to choose which model is better
# Though RandomForestClassifier is best in this scoring test and train

# Kfold fro normal split
kf1 = KFold(n_splits=3)  # split the dataset ito 3 dataset
# example how it split into 3 test and traing data
for train_index, test_index in kf1.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)

# StratifiedKFold fro classsified split
kf2 = StratifiedKFold(n_splits=5)  # split the dataset ito 3 dataset

score_l = []
score_svm = []
score_rf = []


for train_index, test_index in kf2.split(digits.data, digits.target):
    X_tr, X_test, Y_tr, Y_test = digits.data[train_index], digits.data[
        test_index], digits.target[train_index], digits.target[test_index]
    score_l.append(get_score(LogisticRegression(max_iter=1000),
                             X_tr, Y_tr, X_test, Y_test))
    score_rf.append(get_score(RandomForestClassifier(n_estimators=40),
                    X_tr, Y_tr, X_test, Y_test))
    score_svm.append(get_score(SVC(), X_tr, Y_tr, X_test, Y_test))

print('lg ', np.average(score_l))
print('svm ', np.average(score_svm))
print('rf ', np.average(score_rf))

# as above  we can choose SVM for this model

# same comparision using cross_val_score

score_l = cross_val_score(LogisticRegression(
    max_iter=1000), digits.data, digits.target)
score_svm = cross_val_score(SVC(), digits.data, digits.target)
score_rf = cross_val_score(RandomForestClassifier(
    n_estimators=40), digits.data, digits.target)
print('\nusingcrossvalscrore\nlg ', np.average(score_l))
print('svm ', np.average(score_svm))
print('rf ', np.average(score_rf))
