from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
df = pd.read_csv('spam.csv')
print(df.groupby('Category').describe())
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
v = CountVectorizer()
x_train_count = v.fit_transform(df.Message.values)
x_train, x_test, y_train, y_test = train_test_split(
    x_train_count, df.spam, test_size=.2, random_state=10)


print(x_train_count.toarray()[:3])
model = MultinomialNB()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


emails = ['Hey mohon, can we get together to watch football',
          'Upyo 20% discount']
emails_count = v.transform(emails)
print(model.predict(emails_count))

# using pipeline
x_train, x_test, y_train, y_test = train_test_split(
    df.Message, df.spam, test_size=.2, random_state=10)

clf = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))
print(clf.predict(emails))
