import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('salaries.csv')
# input variable
dinput = df.drop('salary_more_then_100k', axis='columns')
# target variable
target = df['salary_more_then_100k']

# checking
# print(input.head())
# print(target.head())
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

dinput['company_n'] = le_company.fit_transform(dinput['company'])
dinput['job_n'] = le_company.fit_transform(dinput['job'])
dinput['degree_n'] = le_company.fit_transform(dinput['degree'])

# checking
print(dinput)

# droping original column of labeled collumn
di = dinput.drop(['company', 'job', 'degree'], axis='columns')

# checking
# print(di.head)
x_train, x_test, y_train, y_test = train_test_split(
    di,  target, test_size=.2, random_state=2)

# checking
# print(x_train.head)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print('Accuracy', model.score(x_test, y_test))  # is not good
test = model.predict(x_test)
print(test)
print('Amazon computer Masters ', model.predict([[0, 1, 0]]))
