from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
dgt = load_digits()
y = dgt.target
# dg_n = dgt.data[100].reshape(8, 8)

# fig, (axs1, axs2) = plt.subplots(1, 2)


# axs1.matshow(dgt.data[10].reshape(8, 8))
# axs1.set_title(dgt.target[10])
# axs2.matshow(dg_n)
# axs2.set_title(dgt.target[100])
# plt.show()

#
# dataframe
df = pd.DataFrame(dgt.data, columns=dgt.feature_names)
# pixel_0_0 has no value its always 0

scaler = StandardScaler()
# array
x_scale = scaler.fit_transform(df)

print(x_scale)

x_train,  x_test, y_train, y_test = train_test_split(
    x_scale, y, test_size=.2, random_state=2)


ln = LogisticRegression(max_iter=1000)
ln.fit(x_train, y_train)
print(ln.score(x_test, y_test))
# 0.9472222222222222

# as we saw that some features or columns has no impact model so we can reduce these 64 columns
# So that our model can train faster
# here  we need PCA =principle Compnent analysis
# PCA need paramaters how many columns are important to train the model to get target.
# means : which columns impact on the results
# in this example we can say 95% column are important. we give numeric value also like 50 column or 10 column

pc = PCA(n_components=0.95)
x_pca = pc.fit_transform(df)
print(x_pca.shape)
# it reduces 64 to 29 column and transorm it to new featuers columns
# now we train the model with this new features

x_train,  x_test, y_train, y_test = train_test_split(
    x_pca, y, test_size=.2, random_state=2)


ln = LogisticRegression(max_iter=1000)
ln.fit(x_train, y_train)
print(ln.score(x_test, y_test))

# 0.9416666666666667 which almost same as 0.9472222222222222
