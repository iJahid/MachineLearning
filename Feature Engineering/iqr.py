import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('heights.csv')
print(df.describe())

q1 = df.height.quantile(.25)
q3 = df.height.quantile(.75)

iqr = q3-q1

lowerlimit = q1-1.5*iqr
upperlimit = q3+1.5*iqr
df_o = df[((df.height < lowerlimit) | (df.height > upperlimit))]
print(df_o)

print('f\n', df[((df.height > lowerlimit) & (df.height < upperlimit))])

# exercise
df = pd.read_csv('height_weight.csv')
print(df.describe())

plt.hist(df[['weight']], bins=40, rwidth=0.8, density=True)
plt.show()
plt.hist(df[['height']], bins=40, rwidth=0.8, density=True)
plt.show()

q1, q3 = df.height.quantile([.25, .75])
iqr = q3-q1.v     
lowerlimit, upperlimit = q1-1.5*iqr, q3+1.5*iqr
print('height no outliar ', lowerlimit, upperlimit, '\n',
      df[((df.height > lowerlimit) & (df.height < upperlimit))])

q1, q3 = df.weight.quantile([.25, .75])

lowerlimit, upperlimit = q1-1.5*iqr, q3+1.5*iqr
print('weight no outliar ', lowerlimit, upperlimit, '\n',
      df[((df.weight > lowerlimit) & (df.weight < upperlimit))])
