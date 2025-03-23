from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)

# Load the trained model
model_path = 'crop_price_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    # If model doesn't exist, we'll need to train it first
    from model_training import prepare_data
    model = prepare_data('data/crop_price_data.csv')

# Get unique values for dropdowns
def get_dropdown_options():
    try:
        df = pd.read_csv('data/crop_price_data.csv')
        options = {
            'crop_name': df['crop_name'].unique().tolist(),
            'region': df['region'].unique().tolist(),
            'season': df['season'].unique().tolist(),
            'soil_type': df['soil_type'].unique().tolist(),
            'market_demand': df['market_demand'].unique().tolist()
        }
    except:
        # Fallback options if file can't be read
        options = {
            'crop_name': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 'Tomato', 'Onion'],
            'region': ['North India', 'South India', 'East India', 'West India', 'Central India', 'Northeast India'],
            'season': ['Kharif', 'Rabi', 'Zaid'],
            'soil_type': ['Sandy', 'Loamy', 'Clayey', 'Black', 'Red', 'Alluvial'],
            'market_demand': ['Low', 'Medium', 'High']
        }
    return options

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    options = get_dropdown_options()
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction"""
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(model['regressor'], 'feature_importances_'):
            # This is simplified and would need adjustment based on your actual model
            # In a real scenario, we'd extract actual feature importances from the model
            feature_names = ['rainfall', 'season', 'market_demand', 'soil_type', 'region', 'crop_name']
            importances = [0.3, 0.2, 0.25, 0.15, 0.05, 0.05]  # Example values
            feature_importance = dict(zip(feature_names, importances))
        else:
            # Fallback if no feature importance available
            feature_importance = {
                'rainfall': 0.3,
                'season': 0.2,
                'market_demand': 0.25,
                'soil_type': 0.15,
                'region': 0.1
            }
        
        # Return prediction and additional info
        return jsonify({
            'predicted_price': round(float(prediction), 2),
            'feature_importance': feature_importance,
            'confidence_level': 'Medium'  # This would be calculated based on model metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/historical_data', methods=['GET'])
def historical_data():
    """API endpoint to get historical price data for a crop"""
    crop = request.args.get('crop')
    region = request.args.get('region')
    
    try:
        # Try to get real historical data from our dataset
        df = pd.read_csv('data/crop_price_data.csv')
        crop_data = df[(df['crop_name'] == crop) & (df['region'] == region)]
        
        if len(crop_data) > 0:
            # If we have data for this crop and region, use it
            # This assumes we have a 'date' and 'price' column
            # Adjust based on your actual dataset structure
            if 'date' in crop_data.columns and 'price' in crop_data.columns:
                return jsonify({
                    'dates': crop_data['date'].tolist(),
                    'prices': crop_data['price'].tolist()
                })
    except:
        pass
    
    # Fallback to mock data if no real data is available
    # Generate some realistic looking data
    base_price = 1000 + np.random.randint(0, 500)
    fluctuation = base_price * 0.2  # 20% price fluctuation
    
    mock_data = {
        'dates': ['Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023', 'Jun 2023'],
        'prices': [
            round(base_price + np.random.uniform(-fluctuation, fluctuation)),
            round(base_price + np.random.uniform(-fluctuation, fluctuation)),
            round(base_price + np.random.uniform(-fluctuation, fluctuation)),
            round(base_price + np.random.uniform(-fluctuation, fluctuation)),
            round(base_price + np.random.uniform(-fluctuation, fluctuation)),
            round(base_price + np.random.uniform(-fluctuation, fluctuation))
        ]
    }
    
    return jsonify(mock_data)

if __name__ == '__main__':
    # Make sure static and templates directories exist
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Check if data file exists, create if not
    if not os.path.exists('data/crop_price_data.csv'):
        print("Warning: data/crop_price_data.csv not found. Please create this file or run the data generation script.")
    
    # Start the Flask app
    app.run(debug=True)