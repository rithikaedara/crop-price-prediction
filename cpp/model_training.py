import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_data(filepath):
    """
    Prepare and preprocess the crop price dataset
    """
    print(f"Loading data from {filepath}...")
    
    # Load data
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating a synthetic dataset for demonstration...")
        df = create_synthetic_dataset()
        df.to_csv(filepath, index=False)
        print(f"Synthetic dataset created and saved to {filepath}")
    
    # Check for required columns
    required_columns = ['crop_name', 'region', 'season', 'rainfall', 'soil_type', 'market_demand', 'price']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing required columns: {missing_columns}")
        print("Creating a synthetic dataset with the correct structure...")
        df = create_synthetic_dataset()
        df.to_csv(filepath, index=False)
        print(f"Synthetic dataset created and saved to {filepath}")
    
    # Handle missing values
    print("Preprocessing data...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values.")
        print("Filling missing values...")
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Define feature columns
    categorical_features = ['crop_name', 'region', 'season', 'soil_type', 'market_demand']
    numerical_features = ['rainfall']
    
    # Target variable
    y = df['price']
    
    # Features
    X = df.drop('price', axis=1)
    
    # Drop any columns that aren't in our features list
    valid_columns = categorical_features + numerical_features
    X = X[valid_columns]
    
    print(f"Training model with {len(X)} samples...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing for numerical features
    numerical_transformer = StandardScaler()
    
    # Define preprocessing for categorical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Feature importance analysis
    try:
        os.makedirs('static/img', exist_ok=True)
        if hasattr(model['regressor'], 'feature_importances_'):
            # Get feature names after one-hot encoding
            categorical_features_transformed = []
            for cat_feature in categorical_features:
                categories = np.unique(X[cat_feature])
                for category in categories:
                    categorical_features_transformed.append(f"{cat_feature}_{category}")
            
            # Combine feature names
            feature_names = numerical_features + categorical_features_transformed
            
            # Get feature importances
            importances = model['regressor'].feature_importances_
            
            # Ensure feature names and importances have the same length
            importances = importances[:len(feature_names)] if len(importances) > len(feature_names) else importances
            feature_names = feature_names[:len(importances)] if len(feature_names) > len(importances) else feature_names
            
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features
            top_indices = indices[:10]
            
            plt.figure(figsize=(10, 6))
            plt.title('Top 10 Feature Importance')
            plt.bar(range(len(top_indices)), importances[top_indices], align='center')
            plt.xticks(range(len(top_indices)), [feature_names[i] for i in top_indices], rotation=90)
            plt.tight_layout()
            plt.savefig('static/img/feature_importance.png')
            print("Feature importance plot saved to static/img/feature_importance.png")
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
    
    # Save the model
    model_path = 'crop_price_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    return model

def create_synthetic_dataset(n_samples=1000):
    """
    Create a synthetic dataset for demonstration purposes
    """
    # Define the possible values for categorical features
    crop_names = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 'Tomato', 'Onion']
    regions = ['North India', 'South India', 'East India', 'West India', 'Central India', 'Northeast India']
    seasons = ['Kharif', 'Rabi', 'Zaid']
    soil_types = ['Sandy', 'Loamy', 'Clayey', 'Black', 'Red', 'Alluvial']
    market_demands = ['Low', 'Medium', 'High']
    
    # Create random data
    np.random.seed(42)
    data = {
        'crop_name': np.random.choice(crop_names, n_samples),
        'region': np.random.choice(regions, n_samples),
        'season': np.random.choice(seasons, n_samples),
        'rainfall': np.random.uniform(300, 2000, n_samples),
        'soil_type': np.random.choice(soil_types, n_samples),
        'market_demand': np.random.choice(market_demands, n_samples),
    }
    
    # Create a dataframe
    df = pd.DataFrame(data)
    
    # Generate synthetic prices based on the features
    # Base prices for different crops
    base_prices = {
        'Rice': 1800, 
        'Wheat': 2000,
        'Maize': 1600,
        'Cotton': 5500,
        'Sugarcane': 350,
        'Potato': 1200,
        'Tomato': 1500,
        'Onion': 1800
    }
    
    # Market demand multipliers
    demand_multipliers = {
        'Low': 0.8,
        'Medium': 1.0,
        'High': 1.2
    }
    
    # Season multipliers
    season_multipliers = {
        'Kharif': 0.95,
        'Rabi': 1.05,
        'Zaid': 1.1
    }
    
    # Calculate prices
    prices = []
    for i in range(n_samples):
        base_price = base_prices[df.iloc[i]['crop_name']]
        demand_mult = demand_multipliers[df.iloc[i]['market_demand']]
        season_mult = season_multipliers[df.iloc[i]['season']]
        
        # Add some rainfall effect (more rain, lower price due to higher supply)
        rainfall = df.iloc[i]['rainfall']
        rainfall_effect = 1.0 - (rainfall - 1000) / 3000  # Normalize around 1000mm
        rainfall_effect = max(0.8, min(1.2, rainfall_effect))  # Constrain effect
        
        # Calculate final price with some random noise
        price = base_price * demand_mult * season_mult * rainfall_effect
        price = price * np.random.uniform(0.9, 1.1)  # Add some noise
        
        prices.append(round(price, 2))
    
    df['price'] = prices
    
    # Add dates for historical data (last 6 months)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    dates = []
    for _ in range(n_samples):
        month = np.random.choice(months)
        dates.append(f"{month} 2023")
    
    df['date'] = dates
    
    return df

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Define the filepath
    filepath = 'data/crop_price_data.csv'
    
    # Prepare data and train model
    model = prepare_data(filepath)
    
    print("Model training complete!")