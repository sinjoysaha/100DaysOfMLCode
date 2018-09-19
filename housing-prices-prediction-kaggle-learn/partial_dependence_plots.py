import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.preprocessing import Imputer

data = pd.read_csv('train.csv')
print(data.columns)
cols_to_use = ['LotArea', 'YearBuilt', 'GarageArea']


def get_some_data():
  y = data.SalePrice
  X = data[cols_to_use]
  my_imputer = Imputer()
  imputed_X = my_imputer.fit_transform(X)
  return imputed_X, y


X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model, features=[0, 2], X=X, feature_names=cols_to_use, grid_resolution=10)
