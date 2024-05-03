import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

to_predict = 10

digits = load_digits()
# plt.matshow(digits.images[to_predict])
# plt.show()
df = pd.DataFrame(digits.data)


X_tr, X_test, Y_tr, Y_test = train_test_split(
    df, digits.target, test_size=.2, random_state=10)

df['target'] = digits.target
print(len(X_test.values[0]))

to_predict = 98

model = RandomForestClassifier(
    n_estimators=200)  # number of trees
model.fit(X_tr, Y_tr)
print(model.score(X_test, Y_test))
y_perdict = model.predict([X_test.values[to_predict]])

print('Predicted ', y_perdict, 'acutal :', Y_test[to_predict])
y_perdict = model.predict(X_test)
cm = confusion_matrix(Y_test, y_perdict)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
