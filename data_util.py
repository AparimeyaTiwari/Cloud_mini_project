import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import random

def generate_sample_data(n_samples=1000):
    """Generate a sample dataset for training models."""
    np.random.seed(42)
    
    # Generate random data
    data = {
        'age': np.random.randint(18, 65, n_samples),
        'city_type': np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'], n_samples),
        'occupation': np.random.choice(['Professional', 'Business', 'Student', 'Other'], n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
        'rent': np.random.randint(5000, 50000, n_samples),
        'loan_repayment': np.random.randint(0, 20000, n_samples),
        'insurance': np.random.randint(1000, 10000, n_samples),
        'groceries': np.random.randint(2000, 20000, n_samples),
        'transport': np.random.randint(1000, 10000, n_samples),
        'eating_out': np.random.randint(1000, 15000, n_samples),
        'entertainment': np.random.randint(1000, 10000, n_samples),
        'utilities': np.random.randint(1000, 8000, n_samples),
        'healthcare': np.random.randint(1000, 10000, n_samples)
    }
    
    # Calculate disposable income
    data['disposable_income'] = data['income'] - (
        data['rent'] + data['loan_repayment'] + data['insurance'] +
        data['groceries'] + data['transport'] + data['eating_out'] +
        data['entertainment'] + data['utilities'] + data['healthcare']
    )
    
    return pd.DataFrame(data)

def load_data():
    """Load or generate the dataset."""
    try:
        # Try to load existing data
        data = pd.read_csv('data.csv')
    except:
        # Generate new data if file doesn't exist
        data = generate_sample_data()
        data.to_csv('data.csv', index=False)
    
    return data

def train_models(data):
    """Train ML models on the dataset."""
    # Encode categorical variables
    le_city = LabelEncoder()
    le_occupation = LabelEncoder()
    
    data['city_type_encoded'] = le_city.fit_transform(data['city_type'])
    data['occupation_encoded'] = le_occupation.fit_transform(data['occupation'])
    
    # Prepare features for clustering
    features_for_clustering = [
        'age', 'city_type_encoded', 'occupation_encoded',
        'income', 'dependents', 'rent', 'loan_repayment',
        'insurance', 'groceries', 'transport', 'eating_out',
        'entertainment', 'utilities', 'healthcare'
    ]
    
    X_cluster = data[features_for_clustering]
    
    # Train KMeans model for spending type
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['spending_type'] = kmeans.fit_predict(X_cluster)
    
    # Prepare features for regression
    features_for_regression = [
        'age', 'city_type_encoded', 'occupation_encoded',
        'income', 'dependents', 'rent', 'loan_repayment',
        'insurance', 'groceries', 'transport', 'eating_out',
        'entertainment', 'utilities', 'healthcare'
    ]
    
    X_reg = data[features_for_regression]
    y_reg = data['disposable_income']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_reg, y_reg)
    
    return kmeans, lr, le_city, le_occupation

def predict_disposable_income(model, le_city, le_occupation, user_data):
    """Predict disposable income for a user."""
    # Prepare user data
    user_df = pd.DataFrame([user_data])
    
    # Encode categorical variables
    user_df['city_type_encoded'] = le_city.transform(user_df['city_type'])
    user_df['occupation_encoded'] = le_occupation.transform(user_df['occupation'])
    
    # Prepare features
    features = [
        'age', 'city_type_encoded', 'occupation_encoded',
        'income', 'dependents', 'rent', 'loan_repayment',
        'insurance', 'groceries', 'transport', 'eating_out',
        'entertainment', 'utilities', 'healthcare'
    ]
    
    X = user_df[features]
    
    # Make prediction
    return model.predict(X)[0]

def get_spending_type(model, le_city, le_occupation, user_data):
    """Determine spending type for a user."""
    # Prepare user data
    user_df = pd.DataFrame([user_data])
    
    # Encode categorical variables
    user_df['city_type_encoded'] = le_city.transform(user_df['city_type'])
    user_df['occupation_encoded'] = le_occupation.transform(user_df['occupation'])
    
    # Prepare features
    features = [
        'age', 'city_type_encoded', 'occupation_encoded',
        'income', 'dependents', 'rent', 'loan_repayment',
        'insurance', 'groceries', 'transport', 'eating_out',
        'entertainment', 'utilities', 'healthcare'
    ]
    
    X = user_df[features]
    
    # Get cluster
    cluster = model.predict(X)[0]
    
    # Map cluster to spending type
    spending_types = {
        0: "Conservative",
        1: "Moderate",
        2: "Liberal"
    }
    
    return spending_types[cluster] 