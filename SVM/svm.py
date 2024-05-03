from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris = load_iris()
# besic featurs
# ['DESCR', 'data', 'data_module', 'feature_names'=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
# 'filename', 'frame', 'target', 'target_names'=['setosa' 'versicolor' 'virginica']]
# print(iris.feature_names)


df = pd.DataFrame(iris.data, columns=iris.feature_names)
# adding target cplumn
df['target'] = iris.target
# print(df.head)
# adding flowername in the dataframe mappping target column to target_name
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],
            color="green", marker='+')
plt.scatter(df1['sepal length (cm)'],
            df1['sepal width (cm)'], color="blue", marker='.')
plt.scatter(df2['sepal length (cm)'],
            df2['sepal width (cm)'], color="red", marker='o')
plt.show()

plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],
            color="green", marker='+')
plt.scatter(df1['petal length (cm)'],
            df1['petal width (cm)'], color="blue", marker='.')
plt.scatter(df2['petal length (cm)'],
            df2['petal width (cm)'], color="red", marker='o')
plt.show()

# so petal classification is more accurate

# spliting for training and testing
X = df.drop(['target', 'flower_name'], axis='columns')
Y = df.target


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=.2, random_state=10)

print(x_test.head)

model = SVC(C=5, gamma=100, random_state=10,
            degree=10)  # C is rgularizaiton line
model.fit(x_train, y_train)
print('accuracy ', model.score(x_test, y_test))  # 53%

model = SVC(C=1, gamma=1, random_state=10,
            degree=10)  # C is rgularizaiton line
model.fit(x_train, y_train)
print('accuracy ', model.score(x_test, y_test))  # 96%
