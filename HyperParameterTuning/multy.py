import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
model_param = {
    'svm': {
        'model': SVC(gamma='auto'), 'params': {
            'C': [1, 3, 5, 10, 20], 'kernel': ['linear', 'rbf']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 3, 4, 10]
        }
    },
    'logisting_reg': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 4, 6]
        }

    }


}

# plt.imshow(mpimg.imread('iris.png'))
# plt.show()
iris = load_iris()
score = []
for model_name, mp in model_param.items():
    clf = GridSearchCV(mp['model'], mp['params'],
                       cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    score.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_param': clf.best_params_
    })
# df = pd.DataFrame(clf.cv_results_)
# print(df[['param_C', 'param_kernel', 'mean_test_score']])
# print('best score', clf.best_score_, 'best param', clf.best_params_)

df = pd.DataFrame(score)
print(df)
