import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import norm
import numpy as np

df = pd.read_csv('weight-height.csv')
df = df[['Gender', 'Height']]
print(df.shape)
print(df.describe())

# matplotlib.rcParams["figure.figsize"] = (10, 6)

# histogram
# plt.hist(df.Height, bins=40, rwidth=0.8, density=True)
# plt.xlabel('height Inch')
# plt.ylabel('count')
# rng = np.arange(df.Height.min(), df.Height.max(), 0.1)
# plt.plot(rng, norm.pdf(rng, df.Height.mean(), df.Height.std()))
# plt.show()

# 3 standard deviation

upperlimit = df.Height.mean()+3 * df.Height.std()
lowerlimit = df.Height.mean()-3 * df.Height.std()
print(upperlimit, lowerlimit)

# let see these outliar
df_o = df[(df.Height > upperlimit) | (df.Height < lowerlimit)]

print('outliar 3 std ', df_o.shape, '\n', df_o)

# only 7 data which we can discuss it with business manager should we keep it?
# cause statistical purpose we can remove these it will help the model do perfection
df_f = df[(df.Height < upperlimit) & (df.Height > lowerlimit)]
print('valid data 3 std', df_f.shape)

# now zscore = (datapoint -mean) /std dev
# it is same as previous 3 standard deviation with upper limit and lower limit...
# zscore is the faster way to do the 3 standard deviation

df['zscore'] = (df.Height-df.Height.mean())/df.Height.std()

# print(df.head(10))

df_z = df[(df.zscore > -3) & (df.zscore < 3)]
print('valid data zscore', df_z.shape)

# same as df_f 9993

df_z_o = df[(df.zscore < -3) | (df.zscore > 3)]
print('outliar zscorw', df_z_o.shape, '\n', df_z_o)
