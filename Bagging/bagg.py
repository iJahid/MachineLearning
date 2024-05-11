from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetes.csv')

# Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
X = df.drop('Outcome', axis='columns')
Y = df.Outcome

# print(df.isnull().sum())

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=.2, random_state=30)
# checking  if the ratio of training and original data are nearby
# cause if every testing data has 'outcome' field 0 how it will calcluate the 'outcome' of 1
# so 'Outcome' ratio in  original and training have to be exact or neary by
# in this data 0 1 ratio in main csv is  0.536  and  in training 0.55  which is almost near
vl_c = df.Outcome.value_counts()
vl_y = y_train.value_counts()
ratio_c = vl_c[1]/vl_c[0]
ratio_y = vl_y[1]/vl_y[0]


print(ratio_c, ratio_y)

# now lets try with scaler (standard)
scaler = StandardScaler()
x_scale = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(
    x_scale, Y, test_size=.2, random_state=30)
vl_y = y_train.value_counts()
ratio_y = vl_y[1]/vl_y[0]
# checking again ratio = which as previous one excact..
# the data scaled fine
print(ratio_c, ratio_y)


# now model creation and training with original data without scaler
# DecisionTree is an imbalance classifier so we will try with this one
cs = cross_val_score(DecisionTreeClassifier(), X, Y, cv=5)
print("Accuracy :", cs.mean())
# 72%

bg = BaggingClassifier(DecisionTreeClassifier(),
                       n_estimators=100, max_samples=.8, oob_score=True, random_state=0)
bg.fit(x_train, y_train)
print(bg.oob_score_)
print(bg.score(x_test, y_test))

score = cross_val_score(bg, X, Y, cv=5)
print('DecisionTree', score.mean())

# now lets try Random forest
score = cross_val_score(RandomForestClassifier(), X, Y, cv=5)
print('Randomforest', score.mean())
