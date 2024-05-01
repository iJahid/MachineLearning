from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import pickle as pkl
from sklearn.linear_model import LinearRegression as lnr
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('homeprices.csv')
print(df)

# create dummy variable
dummy = (pd.get_dummies(df.town, dtype=int))
merged = (pd.concat([df, dummy], axis='columns'))
final = merged.drop(['town'], axis='columns')
print(final)

X = final.drop(['price'], axis='columns')
print(X)
Y = final.price
print(Y)
model = lnr()
model.fit(X.values, Y)
# test with original
print('Accuracy:', model.score(X, Y)*100, '%')
# predict robinsville 3000 sqft price where 2900=600K and 3100=620K already exist
to_predict = pd.DataFrame([[3000, 0, 1, 0]])
print(model.predict(to_predict.values))
