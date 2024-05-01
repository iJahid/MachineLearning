import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sbintance
import math
df = pd.read_csv("data_multi.csv")
print(df.shape)
print(df.describe)

# NaN need to be fill with median

median_bed = math.floor(df.bed.median())
print(median_bed)

df.bed = df.bed.fillna(median_bed)
print(df.describe)

plt.scatter(df.area, df.price, color='red', marker='+')

plt.show()

regr = LinearRegression()

regr.fit(df[['area', 'bed', 'age']], df.price)

# inerceptor
print("Interceptor", regr.intercept_)
print("Coeffecient", regr.coef_)

# single prediction
price = regr.predict([[2500, 4, 15]])
print('2500 sqft, 4 beds, 15 years old Price : ', price)

# prediction for another dataset
df2 = pd.read_csv("data_multi_predict.csv")
print(df2.describe)
price = regr.predict(df2[['area', 'bed', 'age']])

data1 = {'Actual': df2.price, 'Predict': price.flatten()}
df = pd.DataFrame(data1)
print(df)
