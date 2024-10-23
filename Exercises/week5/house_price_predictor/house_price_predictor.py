import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

# A 
# Predict price based on characteristics such as size, number of rooms, city, age of the house, etc.
# Ability to update the model with new data regularly
# Simple user interface for brokers to enter data and get price predictions

#data = pd.read_csv('path_to_your_data.csv')

num_rows = 150
data = {
    'size': np.random.randint(50, 300, size=num_rows),  # size in square meters
    'rooms': np.random.randint(1, 6, size=num_rows),     # number of rooms
    'city': np.random.choice([1,2,3,4,5], size=num_rows),
    'age': np.random.randint(0, 100, size=num_rows),      # age of the house in years
    'price': np.random.randint(1000000, 10000000, size=num_rows)  # price in currency units
}
data = pd.DataFrame(data)

data.to_csv('ai-development/Exercises/week5/house_price_predictor/house_data.csv', index=False)

X = data[['size', 'rooms', 'city', 'age']]
y = data['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with open('ai-development/Exercises/week5/house_price_predictor/house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)

app = Flask(__name__)

with open('ai-development/Exercises/week5/house_price_predictor/house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    size = float(request.form['size'])
    rooms = int(request.form['rooms'])
    city = int(request.form['city'])
    age = float(request.form['age'])

    # Prepare the input for the model
    input_data = pd.DataFrame([[size, rooms, city, age]], columns=['size', 'rooms', 'city', 'age'])
    
    # Predict price
    prediction = model.predict(input_data)
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/update_model', methods=['POST'])
def update_model():
    # Get new data from the request
    new_data = request.get_json()
    new_data_df = pd.DataFrame(new_data)

    # Load existing data (assuming you have saved it previously)
    existing_data = pd.read_csv('ai-development/Exercises/week5/house_price_predictor/house_data.csv')

    # Append new data to existing data and save the new original data
    updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)
    updated_data.to_csv('ai-development/Exercises/week5/house_price_predictor/house_data.csv', index=False)

    # Define features and target
    X = updated_data[['size', 'rooms', 'city', 'age']]
    y = updated_data['price']

    # Normalize or scale the features if necessary
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Retrain the model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Save the updated model
    with open('ai-development/Exercises/week5/house_price_predictor/house_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return jsonify({'message': 'Model updated successfully'})

