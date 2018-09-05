import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('train.csv')
# data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['Id', 'SalePrice'], axis=1)  # .select_dtypes(exclude=['object'])
low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 8 and X[cname].dtype == "object"]
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
X = X[low_cardinality_cols + numeric_cols]
X = pd.get_dummies(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
my_model = XGBRegressor()
my_model.fit(train_X, train_y)
predictions = my_model.predict(test_X)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# Training on full data and predicting on  actual test data
test = pd.read_csv('test.csv')
test_data = test.drop(['Id'], axis=1)
test_data = test_data[low_cardinality_cols + numeric_cols]
test_data = pd.get_dummies(test_data)

X = X[test_data.columns]

my_imputer_full_data = Imputer()
X = my_imputer_full_data.fit_transform(X)
my_model_full_data = XGBRegressor()
my_model_full_data.fit(X, y)
test_data = my_imputer_full_data.transform(test_data)

predictions = my_model_full_data.predict(test_data)

output = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
