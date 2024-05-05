from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('titanic.csv')
# PassengerId,Name,Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked,Survived
input = df[['Pclass', 'Sex', 'Age', 'Fare']]
target = df.Survived

print(input.head(5))

dummies = pd.get_dummies(input.Sex)
print(dummies.head(3))
input = pd.concat([input, dummies], axis='columns')
print(input.head(3))
input.drop('Sex', axis='columns', inplace=True)
print(input.head(3))
print('if any NA value in dataset', input.columns[input.isna().any()])
# Age column
input.Age = input.Age.fillna(input.Age.mean())

print('if any NA value in dataset', input.columns[input.isna().any()])
# none

x_train, x_test, y_train, y_test = train_test_split(
    input, target, test_size=.2, random_state=10)
print(x_test)

model = GaussianNB()
model.fit(x_train, y_train)
print('Accuracy ', model.score(x_test, y_test))
