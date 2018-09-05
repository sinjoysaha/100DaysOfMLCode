# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


def score_dataset(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(60, random_state=42)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
# home_data = home_data.dropna(axis=1)

# Create target object and call it y
y = home_data.SalePrice

# Create X
X = home_data.drop(['Id', 'SalePrice'], axis=1)

# numeric feartures only
# X = X.select_dtypes(exclude=['object'])

low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 8 and X[cname].dtype == "object"]
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

X = X[low_cardinality_cols + numeric_cols]
X = pd.get_dummies(X)
'''
# Split into validation and training data_X
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=3)

cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]
reduced_train_X = train_X.drop(cols_with_missing, axis=1)
print(reduced_train_X.shape)
reduced_val_X = val_X.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values: " + str(score_dataset(reduced_train_X, reduced_val_X, train_y, val_y)))
print()

my_imputer = Imputer()
imputed_train_X = my_imputer.fit_transform(train_X)
print(imputed_train_X.shape)
imputed_val_X = my_imputer.transform(val_X)
print("Mean Absolute Error from Imputation: " + str(score_dataset(imputed_train_X, imputed_val_X, train_y, val_y)))
print()

imputed_train_X_plus = train_X.copy()
imputed_val_X_plus = val_X.copy()
cols_with_missing = (col for col in train_X.columns if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_train_X_plus[col + '_was_missing'] = imputed_train_X_plus[col].isnull()
    imputed_val_X_plus[col + '_was_missing'] = imputed_val_X_plus[col].isnull()

imputed_train_X_plus = my_imputer.fit_transform(imputed_train_X_plus)
print(imputed_train_X_plus.shape)
imputed_val_X_plus = my_imputer.transform(imputed_val_X_plus)
print("Mean Absolute Error from Imputation while Track What Was Imputed:" + str(score_dataset(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y)))
print()
'''
# On actual test data

test_data_path = 'test.csv'
# read test data file using pandas
test = pd.read_csv(test_data_path)
test_data = test.drop(['Id'], axis=1)
test_data = test_data[low_cardinality_cols + numeric_cols]
test_data = pd.get_dummies(test_data)
my_imputer = Imputer()
imputed_plus_train_X = X.copy()
imputed_plus_test = test_data.copy()

# if you want to use fit_transform on test instead of transform below, uncomment next line to reduce columns in train
#imputed_plus_train_X = imputed_plus_train_X[imputed_plus_test.columns]

imputed_plus_train_X, imputed_plus_test = imputed_plus_train_X.align(imputed_plus_test, join='left', axis=1)

cols_with_missing = (col for col in X.columns if X[col].isnull().any())
for col in cols_with_missing:
    imputed_plus_train_X[col + '_was_missing'] = imputed_plus_train_X[col].isnull()
    imputed_plus_test[col + '_was_missing'] = imputed_plus_test[col].isnull()

imputed_plus_train_X = my_imputer.fit_transform(imputed_plus_train_X)
imputed_plus_test = my_imputer.transform(imputed_plus_test)  # if you want to use fit_transform with test, reduce columns in train above

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(60, random_state=1)
# fit rf_model_on_full_data on all data from the
rf_model_on_full_data.fit(imputed_plus_train_X, y)
test_preds = rf_model_on_full_data.predict(imputed_plus_test)

output = pd.DataFrame({'Id': test.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
