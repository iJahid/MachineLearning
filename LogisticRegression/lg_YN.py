import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('insurance_data.csv')
print(df.head)

plt.scatter(df.age, df.bought_insurance, marker='+', color='blue')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    df[['age']], df.bought_insurance, test_size=.2, random_state=10)
print(x_test)
mdl = LogisticRegression()
mdl.fit(x_train, y_train)

prd = mdl.predict(x_test)
print(prd)

print(mdl.score(x_test, y_test))
print(mdl.predict_proba(x_test))
print(mdl.predict([[48]]))
