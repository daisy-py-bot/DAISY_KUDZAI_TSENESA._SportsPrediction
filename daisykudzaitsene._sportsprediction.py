# -*- coding: utf-8 -*-
"""RatingModel-2-2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12H4LDNUaw7nR_UIkvVVdvHo_lksRvUy0

# Import Libraries
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import pickle
from sklearn.ensemble import RandomForestRegressor
import json

# load the dataset
data = pd.read_csv('male_players+%28legacy%29.csv', low_memory=False)
data.head(2)

data.info()

data.describe()

"""# Data Preprocessing

### Remove identifier columns
"""

# DEFINE A FUNCTION TO REMOVE IDENTIFIER COLUMNS

def remove_identifiers(data, identifier_attributes):
    data.drop(columns= identifier_attributes, inplace = True)

# create a list of identifier columns, names or links to be removed
identifier_columns = [
    "player_id",
    "player_url",
    "fifa_version",
    "fifa_update",
    "fifa_update_date",
    "short_name",
    "long_name",
    "league_id",
    "league_name",
    "club_team_id",
    "club_name",
    "club_position",
    "club_jersey_number",
    "club_loaned_from",
    "club_joined_date",
    "nationality_id",
    "nationality_name",
    "nation_team_id",
    "nation_position",
    "nation_jersey_number",
    "player_face_url"
]

# remove the identifiers
remove_identifiers(data, identifier_columns)

"""### Remove columns with more 30% missing data"""

# DEFINE A FUNCTION TO REMOVE MORE THAN 30% MISSING DATA
def remove_empty_columns(data):
    # set the threshold to 30%
    threshold = int(len(data) * 0.3)

    # get the number of missing values per column
    null_values_per_column = pd.DataFrame(data[data.columns].isnull().sum(), columns=['null_value_count'], index=data.columns)

    # get the columns with missing values that exceed the threshold
    columns_exceed_threshold = null_values_per_column[null_values_per_column['null_value_count']>threshold]

    # drop the columns with missing values that exceed the threshold
    data.drop(columns= columns_exceed_threshold.index, inplace=True)

# remove the columns with more than 30% missing data
remove_empty_columns(data)

data.shape

"""### Impute missing data"""

def impute_missing_data(data):
    # separate the data into categorical and numerical data types
    categorical_columns =  data.select_dtypes(['category', 'object']).columns
    numerical_columns = data.select_dtypes('number').columns


    # create an imputer for categorical data and fill in missing data with the most frequent value
    cat_impute = SimpleImputer(strategy='most_frequent')
    data[categorical_columns] = pd.DataFrame(cat_impute.fit_transform(data[categorical_columns]), columns=categorical_columns, index = data.index)


    # create an instance of the SimpleImputer for numerical data: fill in missing data with the average value
    num_impute = SimpleImputer(strategy='mean')
    data[numerical_columns] = pd.DataFrame(cat_impute.fit_transform(data[numerical_columns]), columns = numerical_columns, index=data.index)

# call the function for imputing numerical data
impute_missing_data(data)

data.shape

"""# Feature Engineering

### Calculate effective positional rating
"""

# define a function that calcutates the effective rating for each position and change the data type for the ratings from object to integers
def calculate_effective_positional_rating(rating):
    return eval(rating) # calculate the effective score for each positional rating

def positional_rating(data, positions):
    # calculate the effective rating for each columns in positions
    for pos in positions:
        data[pos] = data[pos].apply(calculate_effective_positional_rating) # pass in columns with the positional rating

# select the columns with ratings for each position
position_ratings = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk']

# call the function to calculate effective ratings
positional_rating(data, position_ratings)

# view the calculated effective ratings
data[position_ratings].head()

data.shape

"""### Encode Categorical Data"""

def encode_categorical_data(data):
    # Extract the categorical features
    categorical_col = data.select_dtypes(['object', 'category']).columns

    # Encode the categorical values with OneHotEncoder
    ohe = OneHotEncoder(sparse_output=True)  # Use sparse=True for memory efficiency
    encoded_data = ohe.fit_transform(data[categorical_col])

    # Convert encoded_data to DataFrame with the proper column names
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_data,
                                                   columns=ohe.get_feature_names_out(categorical_col),
                                                   index=data.index)

    # Drop the categorical features from data
    data.drop(columns=categorical_col, inplace=True)

    # Concatenate encoded_df with data
    comb_encoded_data = pd.concat([encoded_df, data], axis=1)

    print('...Encoding categorical variables')

    return comb_encoded_data

# call the function for encoding data
data = encode_categorical_data(data)

data.shape

"""# Feature Subset"""

def create_feature_subset(data):
  # get the correlation coeffients between all variables in a dataframe
  corr_matrix = data.corr()

  # extract the correlation coeffients between features and the target(overall rating)
  correlation = pd.DataFrame(corr_matrix['overall'])
  correlation = correlation.rename(columns={'overall': 'corr_coeff'}) # change the name of the column to corr_coeff
  print("\nCorrelation coeffients:\n ", correlation['corr_coeff'].sort_values())

  # set a threshold for the correlation value to 40%
  threshold = 0.4

  # select features that are above the threshold
  important_features = correlation[abs(correlation['corr_coeff']) > threshold] # variables with stronger positive correlation

  # get the array of variables
  important_features = np.array(important_features.index)
  print("\n Features with the strongest correlation:\n", important_features)

  return data[important_features] # return the dataframe with the strongest correlation to the target

data = create_feature_subset(data) # new dataframe with the features showing the features with the greatest correlation to the target
data.columns # view the feature subset with the most important features



"""# Create Data Preprocessing Pipeline"""

# create the function for preprocessing data
def preprocess_data(data, identifier_columns, positions):

  # remove identifier columns
  remove_identifiers(data, identifier_columns)

  # remove attributes with more than 50% missing data
  remove_empty_columns(data)

  # impute missing data
  print('...imputing missing data')
  impute_missing_data(data)

  # calculate effective positional rating
  print('...calculating effective positional ratings')
  positional_rating(data, positions)

  # encode categorical data
  data = encode_categorical_data(data)

  # extract the feature subset
  print('...create feature subset')
  data = create_feature_subset(data)

  print('...Data processing complete')

  return data

"""# Feature Scaling"""

# separate the target from the features
Y = data['overall'] # get the target
X = data.drop(columns = ['overall']) # extract the feautres

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
X = scaler.fit_transform(X)

X.shape

Y.shape

X

"""# Model Training"""

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""### Model Selection"""

def model_training(x_train, x_test, y_train, y_test):
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net': ElasticNet(),
        'Decision Tree': DecisionTreeRegressor(),
        'Bayesian Ridge': BayesianRidge(),
        'SGD Regressor': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    }

    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use MAE as the scoring metric
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=True)

    # Dictionary to store the mean scores for each model
    model_scores = {}


    # Train and evaluate each model with cross-validation
    for name, model in models.items():
        print(f"Model: {name}")
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=mae_scorer)
        mean_score = np.mean(scores) # caluculate the average mean absolute error for all folds
        model_scores[name] = mean_score # add the mae for each model to the dict
        print(f"MAE: Mean = {mean_score:.4f}, Std = {scores.std():.4f}")  # print the mean score and the standard deviation
        print("-" * 32) # print a new line


    # Identify the best model based on the mean MAE score
    best_model_name = min(model_scores, key=model_scores.get)  # Choose model with a lower MAE score
    best_model_score = model_scores[best_model_name]

    print(f"Best Model: {best_model_name}")
    print(f"Best Model Score (MAE): {best_model_score:.4f}")

    return models[best_model_name] # return the best model

# train the different models
best_model = model_training(x_train, x_test, y_train, y_test)



"""### Train Best Model"""

def train_best_model(best_model):
  # Train the best model on the entire training set and evaluate the performance on the test set
  best_model.fit(x_train, y_train)
  y_pred = best_model.predict(x_test)
  test_mae = mean_absolute_error(y_test, y_pred)
  print(f"Test MAE of Best Model: {test_mae:.4f}")

  return test_mae # return the mean absolute error for evaluation

# train the best model and measure its performance
base_mae = train_best_model(best_model)

"""# Hyperameter Tuning"""

def tune_hyperparameter(x_train, x_test, y_train, y_test):

    # Define XGBoost regressor
    xgb = XGBRegressor(random_state=42)

    # Define parameters grid for Grid Search
    param_grid = {
    'n_estimators': [100, 200],           # Number of boosting rounds or trees to build
    'max_depth': [3, 5, 7],               # Maximum depth of a tree
    'learning_rate': [0.01, 0.05, 0.1],   # Step size shrinkage used to prevent overfitting
    'subsample': [0.8, 1.0],              # Fraction of samples used to train each tree
    'colsample_bytree': [0.8, 1.0]        # Fraction of features used to train each tree
    }

    # Perform Grid Search CV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Print best parameters
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Save the best XGBoost model using pickle
    with open('best_xgb_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)

    # Save the best parameters using pickle
    with open('best_xgb_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # Evaluate the model performance on test
    y_pred = grid_search.best_estimator_.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Final XGBoost Model Performance (MAE): {mae:.4f}")

    return grid_search.best_estimator_, best_params

best_xgb_model, best_params = tune_hyperparameter(x_train, x_test, y_train, y_test)

def extract_important_features(best_xgb_model):

  # Extract feature importances
  feature_importances = best_xgb_model.feature_importances_

  # Create a DataFrame to display feature importances
  features = data.drop(columns=['overall']).columns
  importance_df = pd.DataFrame({
      'Feature': features,
      'Importance': feature_importances
  })

  # Sort features by importance
  importance_df = importance_df.sort_values(by='Importance', ascending=False)

  # Display the top features
  print(importance_df.head())  # Displaying the top 5 features

  return importance_df

# call the function for extracting the feature importances
important_features = extract_important_features(best_xgb_model)

# display the features ranked by their importances
important_features



"""# Test With New Data Set"""

# import the new dataset
data = pd.read_csv('players_22.csv', low_memory=False)

# create the identifier columns from the new data set
identifier_col = [
    'sofifa_id',
    'player_url',
    'short_name',
    'long_name',
    'dob',
    'player_face_url',
    'club_logo_url',
    'club_flag_url',
    'nation_logo_url',
    'nation_flag_url'
]

# select the columns with ratings for each position
position_rate = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk']


# extract the feature subset for uniformity
feature_subset = ['potential', 'value_eur', 'wage_eur', 'age',
       'international_reputation', 'shooting', 'passing', 'dribbling',
       'physic', 'attacking_short_passing', 'skill_curve',
       'skill_long_passing', 'skill_ball_control', 'movement_reactions',
       'power_shot_power', 'power_long_shots', 'mentality_vision',
       'mentality_composure', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
       'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm',
       'cdm', 'rdm', 'rwb', 'lb', 'rb']

# preprocess that data
data = preprocess_data(data, identifier_col, position_rate)

data.shape

# Function to load the saved model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

# Function to test and evaluate the model
def test_model(data):
    # Separate the target from the features
    Y = data['overall']  # Get the target
    X = data[feature_subset]  # Extract the features

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform the training data
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to a pickle file
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print('...Scaler saved')

    # Paths to saved model and parameters
    model_path = 'best_xgb_model.pkl'
    params_path = 'best_xgb_params.pkl'

    # Load the model and parameters
    best_xgb_model = load_model(model_path)
    best_params = load_model(params_path)

    # Make predictions using the loaded model
    y_pred = best_xgb_model.predict(X_scaled)

    # Measure performance using Mean Absolute Error (MAE)
    mae = mean_absolute_error(Y, y_pred)
    print(f"XGBoost Model Performance on New Data (MAE): {mae:.4f}")

    # Create a default values for model prediction: Calculate the mean for all features
    default_values = X.mean().to_dict()

    # Save the default values to a JSON file
    with open('default_values.json', 'w') as json_file:
        json.dump(default_values, json_file)
    print('... Default values for features created')


# Test and evaluate the model
test_model(data)

