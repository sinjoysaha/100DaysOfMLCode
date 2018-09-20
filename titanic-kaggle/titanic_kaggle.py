import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

data = pd.read_csv('train.csv')
data = data.drop(['PassengerId', 'Name', 'Age', 'Cabin'], axis=1)
y = data['Survived']
X = data.drop(['Survived'], axis=1)
X = pd.get_dummies(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

mymodel = LogisticRegression()
mymodel.fit(train_X, train_y)
prediction = mymodel.predict(test_X)
mae = mean_absolute_error(test_y, prediction)
print(mae)

test_data = pd.read_csv('test.csv')
test_data = test_data.drop(['Name', 'Age', 'Cabin'], axis=1)
test_data_X = test_data.drop(['PassengerId'], axis=1)
test_data_X = pd.get_dummies(test_data_X)

new_train_X = X
new_train_X, test_data_X = new_train_X.align(test_data_X, join='left', axis=1)

my_imputer = Imputer()
new_train_X = my_imputer.fit_transform(new_train_X)
test_data_X = my_imputer.transform(test_data_X)

# Training on full data
full_data_mymodel = LogisticRegression()
full_data_mymodel.fit(new_train_X, y)
test_prediction = mymodel.predict(test_data_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_prediction})
output.to_csv('submission.csv', index=False)
