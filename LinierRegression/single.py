import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sbintance
df = pd.read_csv("data.csv")
print(df.shape)
print(df.describe)

plt.title = "Area Price"
plt.scatter(df.area, df.price, color='red', marker='+')
plt.xlabel = 'area (sqft)'
plt.ylabel = 'Price" (US$)'
df.plot(x='area', y='price')

# plt.show()


plt.figure(figsize=(15, 10))
plt.tight_layout()
sbintance.displot(df['price'])
# plt.show()

# data spliting
X = df['area'].values.reshape(-1, 1)
Y = df['price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
regr = LinearRegression()

regr.fit(X_train, y_train)

# inerceptor
print("Interceptor", regr.intercept_)
print("Coeffecient", regr.coef_)

y_pred = regr.predict(X_test)
data1 = {'Actual': y_test.flatten(), 'Predict': y_pred.flatten()}
df = pd.DataFrame(data1)
print(df)


df.plot(kind='bar', figsize=(16, 10))
# plt.show()

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
# plt.show()
s = pd.DataFrame({3300})
d = regr.predict(s)
print('Predic of 3000sqft', d)
