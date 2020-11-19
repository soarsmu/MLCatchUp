import pandas as pd

file_path='./kc_house_data.csv'

data=pd.read_csv(file_path)

# this is called X
attribute_taken_for_predictor=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','sqft_living15']
X=data[attribute_taken_for_predictor]
Y=data.price

from sklearn.tree import DecisionTreeRegressor

# Defining the model


# Fit the model
housing_model.fit(X,Y)

DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
