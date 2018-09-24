import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
print(data.head(10))
print(data.columns)


survived = data[data["Survived"] == 1]
died = data[data["Survived"] == 0]
survived["Fare"].plot.hist(alpha=0.5, color='red', bins=30, rwidth=0.8)
data["Fare"].plot.hist(alpha=0.5, color='green', bins=30, rwidth=0.8)
plt.legend(['Survived', 'Total'])
plt.show()

data['Age'] = data["Age"].fillna(-0.5)
data['Age_categories'] = pd.cut(data["Age"], [-1, 0, 5, 12, 18, 35, 60, 100], labels=["Missing", 'Infant', "Child", 'Teenager', "Young_Adult", 'Adult', 'Senior'])

data['Parch_categories'] = pd.cut(data["Parch"], [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10],
                                  labels=["0", '1', '2', '3', '4', '5', '6above'])
data['Fare_categories'] = pd.cut(data["Fare"], [-0.5, 20, 40, 60, 80, 100, 200, 300, 1000],
                                 labels=["20", '40', '60', '80', '100', '200', '300', '300above'])


survived = data[data["Survived"] == 1]
survived["Fare_categories"].value_counts().plot.bar()
plt.show()


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


data = create_dummies(data, "Pclass")
data = create_dummies(data, "Sex")
data = create_dummies(data, "Age_categories")
data = create_dummies(data, "Embarked")
data = create_dummies(data, "SibSp")
data = create_dummies(data, "Parch_categories")
data = create_dummies(data, "Fare_categories")

print(data.columns)

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
           'Age_categories_Missing', 'Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young_Adult', 'Age_categories_Adult',
           'Age_categories_Senior', 'Embarked_C',
           'Embarked_Q', 'Embarked_S', 'SibSp_0', 'SibSp_1', 'SibSp_2',
           'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_categories_0',
           'Parch_categories_1', 'Parch_categories_2', 'Parch_categories_3',
           'Parch_categories_4', 'Parch_categories_5', 'Parch_categories_6above',
           'Fare_categories_20', 'Fare_categories_40', 'Fare_categories_60',
           'Fare_categories_80', 'Fare_categories_100', 'Fare_categories_200', 'Fare_categories_300',
           'Fare_categories_300above']

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
test_data['Age_categories'] = pd.cut(test_data["Age"], [-1, 0, 5, 12, 18, 35, 60, 100], labels=["Missing", 'Infant', "Child", 'Teenager', "Young_Adult", 'Adult', 'Senior'])
test_data['Parch_categories'] = pd.cut(test_data["Parch"], [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10],
                                       labels=["0", '1', '2', '3', '4', '5', '6above'])
test_data['Fare'] = data["Fare"].fillna(32.2)
test_data['Fare_categories'] = pd.cut(test_data["Fare"], [-0.5, 20, 40, 60, 80, 100, 200, 300, 1000],
                                      labels=["20", '40', '60', '80', '100', '200', '300', '300above'])


test_data = create_dummies(test_data, "Pclass")
test_data = create_dummies(test_data, "Sex")
test_data = create_dummies(test_data, "Age_categories")
test_data = create_dummies(test_data, "Embarked")
test_data = create_dummies(test_data, "SibSp")
test_data = create_dummies(test_data, "Parch_categories")
test_data = create_dummies(test_data, "Fare_categories")

print(test_data.columns)

test_data_X = test_data[columns]

# Training on full data
full_data_mymodel = LogisticRegression()
full_data_mymodel.fit(X, y)
test_prediction = mymodel.predict(test_data_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_prediction})
output.to_csv('submission.csv', index=False)
