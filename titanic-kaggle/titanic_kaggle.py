import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
print(data.head(10))
print(data.columns)

print(data.Parch.value_counts())
data.pivot_table(index="Parch", values="Survived").plot.bar()
plt.show()

data['Age'] = data["Age"].fillna(-0.5)
data['Age_categories'] = pd.cut(data["Age"], [-1, 0, 5, 12, 18, 35, 60, 100], labels=["Missing", 'Infant', "Child", 'Teenager', "Young_Adult", 'Adult', 'Senior'])

data['Parch_categories'] = pd.cut(data["Parch"], [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10],
                                  labels=["0", '1', '2', '3', '4', '5', '6above'])


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

print(data.columns)

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
           'Age_categories_Missing', 'Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young_Adult', 'Age_categories_Adult',
           'Age_categories_Senior', 'Embarked_C',
           'Embarked_Q', 'Embarked_S', 'SibSp_0', 'SibSp_1', 'SibSp_2',
           'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_categories_0', 'Parch_categories_1',
           'Parch_categories_2', 'Parch_categories_3', 'Parch_categories_4', 'Parch_categories_5', 'Parch_categories_6above']

y = data['Survived']
X = data[columns]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

mymodel = LogisticRegression()
mymodel.fit(train_X, train_y)
prediction = mymodel.predict(test_X)
acc = accuracy_score(test_y, prediction)
print(acc)


test_data = pd.read_csv('test.csv')
print(test_data.Parch.value_counts())
test_data['Age'] = test_data["Age"].fillna(-0.5)
test_data['Age_categories'] = pd.cut(test_data["Age"], [-1, 0, 5, 12, 18, 35, 60, 100], labels=["Missing", 'Infant', "Child", 'Teenager', "Young_Adult", 'Adult', 'Senior'])
test_data['Parch_categories'] = pd.cut(test_data["Parch"], [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10],
                                       labels=["0", '1', '2', '3', '4', '5', '6above'])

test_data = create_dummies(test_data, "Pclass")
test_data = create_dummies(test_data, "Sex")
test_data = create_dummies(test_data, "Age_categories")
test_data = create_dummies(test_data, "Embarked")
test_data = create_dummies(test_data, "SibSp")
test_data = create_dummies(test_data, "Parch")
test_data = create_dummies(test_data, "Parch_categories")

print(test_data.columns)

test_data_X = test_data[columns]

# Training on full data
full_data_mymodel = LogisticRegression()
full_data_mymodel.fit(X, y)
test_prediction = mymodel.predict(test_data_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_prediction})
output.to_csv('submission.csv', index=False)
