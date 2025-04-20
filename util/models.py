import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def load_data():
    # Create a sample dataset with realistic patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base income based on occupation and city type
    base_incomes = {
        'Professional': {'Tier_1': 8000, 'Tier_2': 6000, 'Tier_3': 5000},
        'Self_Employed': {'Tier_1': 7000, 'Tier_2': 5000, 'Tier_3': 4000},
        'Student': {'Tier_1': 2000, 'Tier_2': 1500, 'Tier_3': 1000},
        'Retired': {'Tier_1': 4000, 'Tier_2': 3000, 'Tier_3': 2500}
    }
    
    data = []
    for _ in range(n_samples):
        age = np.random.randint(18, 70)
        city_type = np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'])
        occupation = np.random.choice(['Professional', 'Self_Employed', 'Student', 'Retired'])
        
        # Base income with some variation
        base_income = base_incomes[occupation][city_type]
        income = np.random.normal(base_income, base_income * 0.2)
        
        # Generate expenses based on income
        expenses = {
            'rent': income * 0.3 * np.random.normal(1, 0.1),
            'groceries': income * 0.2 * np.random.normal(1, 0.1),
            'transport': income * 0.1 * np.random.normal(1, 0.1),
            'entertainment': income * 0.1 * np.random.normal(1, 0.1),
            'utilities': income * 0.05 * np.random.normal(1, 0.1),
            'miscellaneous': income * 0.1 * np.random.normal(1, 0.1)
        }
        
        row = {
            'age': age,
            'city_type': city_type,
            'occupation': occupation,
            'income': income,
            **expenses
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)
    return df

def train_kmeans(data):
    # Prepare data for clustering
    expense_columns = ['rent', 'groceries', 'transport', 'entertainment', 'utilities', 'miscellaneous']
    X = data[expense_columns]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Evaluate clustering performance
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)

    return kmeans, scaler, cluster_labels, silhouette_avg



def train_linear_regression(data):
    # Define features for prediction
    categorical_features = ['city_type', 'occupation']
    numerical_features = ['age', 'income']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    X = data[numerical_features + categorical_features]
    X_transformed = preprocessor.fit_transform(X)
    
    # Train separate models for each expense category
    expense_columns = ['rent', 'groceries', 'transport', 'entertainment', 'utilities', 'miscellaneous']
    
    models = {}
    r2_scores = {}
    
    for expense in expense_columns:
        y = data[expense]
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2_scores[expense] = r2_score(y_test, y_pred)
        models[expense] = model
    
    return models, preprocessor, r2_scores

    
    return models, preprocessor

def get_spending_analysis(data, user_data, kmeans, scaler):
    expense_columns = ['rent', 'groceries', 'transport', 'entertainment', 'utilities', 'miscellaneous']
    
    user_expenses = np.array([user_data[col] for col in expense_columns])
    scaled_user_expenses = scaler.transform([user_expenses])
    user_cluster = kmeans.predict(scaled_user_expenses)[0]
    
    data['cluster'] = kmeans.labels_
    cluster_data = data[data['cluster'] == user_cluster]
    
    cluster_stats = cluster_data[expense_columns].mean()
    overspending = {}

    for col in expense_columns:
        user_val = user_data[col]
        avg_val = cluster_stats[col]
        if user_val > avg_val * 1.1:  # more than 10% higher
            overspending[col] = {
                'user': user_val,
                'average': avg_val,
                'difference': user_val - avg_val
            }

    cluster_info = {
        'cluster': user_cluster,
        'cluster_size': len(cluster_data),
        'cluster_stats': cluster_stats.to_dict()
    }

    return None, overspending, cluster_info
