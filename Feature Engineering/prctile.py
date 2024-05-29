import pandas as pd

df = pd.read_csv('AB_NYC_2019.csv')
df1 = (df[['name', 'minimum_nights', 'price']])
print(df1.describe())
mint, maxt = df1.price.quantile([.01, .999])

df2 = df[(df.price < maxt) & (df.price > mint)]
print(df.shape, df2.shape)
print(df2.sample)
