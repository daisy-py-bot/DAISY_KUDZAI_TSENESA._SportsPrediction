from flask import Flask, render_template, request
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load the trained XGBoost model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model

# Path to your trained model
model_path = 'best_xgb_model.pkl'
model = load_model(model_path)

# Load the default values from the JSON file
with open('default_values.json', 'r') as json_file:
    default_values = json.load(json_file)

# Define the feature subset
feature_subset = list(default_values.keys())

# Define the prediction route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve and parse form data
    user_input = {
        'value_eur': float(request.form['value_eur']),
        'movement_reactions': float(request.form['movement_reactions']),
        'age': int(request.form['age']),
        'international_reputation': float(request.form['international_reputation']),
        'wage_eur': float(request.form['wage_eur']),
    }

    # Create a complete feature vector with default values
    input_features = np.array([default_values[feature] for feature in feature_subset])

    # Update the feature vector with user-provided values
    for feature, value in user_input.items():
        if feature in feature_subset:
            index = feature_subset.index(feature)
            input_features[index] = value

    # Reshape input_features to match model input shape
    input_features = input_features.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_features)[0]

    
    # Calculate confidence score based on the deviance from the average player rating
    confidence_score = 100.0 - np.abs(prediction - 75) 

    # Format prediction and confidence score
    format_prediction = f'Predicted Rating: {prediction:.2f}'
    format_confidence = f'Confidence Score: {confidence_score:.2f}'

    # Return the predicted result and confidence score
    return render_template('index.html', prediction_text=format_prediction, confidence_text=format_confidence)

if __name__ == '__main__':
    app.run(debug=True, port=5501)
