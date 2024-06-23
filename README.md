# LINK TO YOUTUBE VIDEO:
https://youtu.be/B8Hi7_m-_sM

# Player Rating Prediction Model

## Overview

This project aims to predict the rating of a player based on various features using a machine learning model. 
The model is trained on historical data of players and their ratings, and it can predict the rating of new players given their attributes.

## Model Details

Model Type: XGBoostRegressor
Features Used: ('potential', 'value_eur', 'wage_eur', 'age',
       'international_reputation', 'shooting', 'passing', 'dribbling',
       'physic', 'attacking_short_passing', 'skill_curve',
       'skill_long_passing', 'skill_ball_control', 'movement_reactions',
       'power_shot_power', 'power_long_shots', 'mentality_vision',
       'mentality_composure', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw',
       'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm',
       'cdm', 'rdm', 'rwb', 'lb', 'rb')
Model's performance metrics: Mean Absolute Error

## Dataset

### Training Dataset
Dataset Name: FIFA 23 Complete Player Dataset (males_legacy.csv)
Source: Kaggle - FIFA 23 Complete Player Dataset
Description: This dataset includes historical data of male players and their attributes, including ratings, used to train the player rating prediction model.

### Testing/Evaluation Dataset
Dataset Name: players_22
Source: Kaggle - FIFA 23 Complete Player Dataset
Description: This dataset is used for testing and evaluating the trained player rating prediction model. It contains data of players similar to those in the training dataset but for a different time period or scenario.


## Data Preprocessing
  - Remove Identifier Columns: Columns that are used solely for identification purposes are removed from the dataset.
  
  - Remove Empty Columns: Attributes with more than 30% missing data are removed to ensure data quality.
  
  - Impute Missing Data: Missing data is imputed using appropriate techniques to ensure completeness of the dataset.
  
  - Calculate Effective Positional Ratings: Effective ratings for different positions (ls, st, rs, etc.) are calculated based on provided ratings.
  
  - Encode Categorical Data: Categorical data is encoded into numerical format to be suitable for machine learning algorithms.

## Feature Engineering

Feature Subset
The selected feature subset includes the most relevant features that correlate strongly with the overall player rating.

## Model Training

## Model Selection
Different regression models were evaluated to select the best performer based on Mean Absolute Error (MAE).

## Hyperparameter Tuning
Hyperparameters for the XGBoost model were tuned using Grid Search Cross Validation to optimize performance.

## Model Evaluation
The best model was evaluated using the FIFA 22 Player Dataset (players_22.csv) to assess its performance in predicting player ratings for new data.

## Web Application
The model is deployed as a web application using Flask. Users can interact with the model through a web interface to predict player ratings. 
The application uses the trained XGBoost model to provide predictions and confidence scores.

Features:
- Input form to enter player attributes
- Display of predicted player rating based on input
- Running the Web Application

  
Clone the repository:

git clone https://github.com/your_username/player-rating-prediction.git](https://github.com/daisy-py-bot/DAISY_KUDZAI_TSENESA._SportsPrediction.git
cd DAISY_KUDZAI_TSENESA._SportsPrediction

Install dependencies:
pip install -r requirements.txt

Run the Flask application:
python app.py

Open a web browser and go to http://127.0.0.1:5501 to use the web application.




