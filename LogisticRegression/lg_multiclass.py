import pandas as pd
from matplotlib import pyplot as plt
# dataset made by 8x8 pxl of 1797 images of digits
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
import seaborn as sbn
digit = load_digits()
# dir(digit)) =['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']

# digit.data[0]=[ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
# 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]

# showing image in the digit using plot
plt.gray()

plt.matshow(digit.images[67])
plt.show()

print('target', digit.target[19])

x_train, x_test, y_train, y_test = train_test_split(
    digit.data,  digit.target, test_size=.2, random_state=10)
print(len(x_train))
mdl = LogisticRegression(solver='lbfgs', max_iter=1000)
mdl.fit(x_train, y_train)
print(mdl.score(x_test, y_test))

predicted_image_index = mdl.predict([digit.data[67]])
print(predicted_image_index)
print(digit.target[predicted_image_index])

# cosfusion Matrix for check existing with prediciton
y_predict = mdl.predict(x_test)
d = confusion_matrix(y_test, y_predict)
print(d)
plt.figure(figsize=(10, 7))
sbn.heatmap(d, annot=True)
plt.xlabel = "Predicted"
plt.ylabel = "Actual"
plt.show()
