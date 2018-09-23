import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data['Age'] = data["Age"].fillna(-0.5)
data['Age_categories'] = pd.cut(data["Age"], [-1, 0, 5, 12, 18, 35, 60, 100], labels=["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior'])


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


data = create_dummies(data, "Pclass")
data = create_dummies(data, "Sex")
data = create_dummies(data, "Age_categories")

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
           'Age_categories_Missing', 'Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young Adult', 'Age_categories_Adult',
           'Age_categories_Senior']

y = data['Survived']
X = data[columns]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

mymodel = LogisticRegression()
mymodel.fit(train_X, train_y)
prediction = mymodel.predict(test_X)
acc = accuracy_score(test_y, prediction)
print(acc)

test_data = pd.read_csv('test.csv')
test_data['Age'] = test_data["Age"].fillna(-0.5)
test_data['Age_categories'] = pd.cut(test_data["Age"], [-1, 0, 5, 12, 18, 35, 60, 100], labels=["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior'])
test_data = create_dummies(test_data, "Pclass")
test_data = create_dummies(test_data, "Sex")
test_data = create_dummies(test_data, "Age_categories")

test_data_X = test_data[columns]
print(test_data_X.columns)
# Training on full data
full_data_mymodel = LogisticRegression()
full_data_mymodel.fit(X, y)
test_prediction = mymodel.predict(test_data_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_prediction})
output.to_csv('submission.csv', index=False)
