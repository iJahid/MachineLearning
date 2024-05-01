from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LinearRegression as lnr
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('homeprices.csv')


# create OneHotEncoded  variable
le = LabelEncoder()
dfle = df
print(dfle)
dfle.town = le.fit_transform(dfle.town)
print(dfle)


X = dfle[['town', 'area']].values
print(X)


y = dfle.price.values

model = lnr()
model.fit(X, y)
print(model.predict([[2, 2800]]))
print(' Accuracy : ', format(round(model.score(X, y) * 100, 2)), "%")
ct = ColumnTransformer([('town', OneHotEncoder(), [0])],
                       remainder='passthrough')


X = ct.fit_transform(X)
print(X)
X = X[:, 1:]  # all column starting from 1
print(X)
model.fit(X, y)
print(model.predict([[0, 1, 2800]]))
print(' Accuracy : ', format(round(model.score(X, y) * 100, 2)), "%")
