"""
Bootstrap script to train and save the model artifact (model.pkl) 
Ensure this is run before starting the API if model.pkl does not exist.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def generate_training_data(num_rows=500):
    np.random.seed(42)
    return pd.DataFrame({
        'city_tier': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], size=num_rows),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45+'], size=num_rows),
        'acquisition_channel': np.random.choice(['Google Search', 'Instagram', 'Influencer', 'Organic'], size=num_rows),
        'loyalty_tier': np.random.choice(['Silver', 'Gold', 'Platinum', 'None'], size=num_rows),
        'preferred_category': np.random.choice(['Skin Care', 'Hair Care', 'Makeup'], size=num_rows),
        'marketing_consent': np.random.choice(['Yes', 'No'], size=num_rows),
        'recency_days': np.random.randint(1, 365, size=num_rows),
        'frequency_180d': np.random.randint(1, 10, size=num_rows),
        'monetary_180d': np.random.uniform(100, 5000, size=num_rows),
        'return_rate_180d': np.random.uniform(0, 1, size=num_rows),
        'avg_discount_pct_180d': np.random.uniform(0, 0.6, size=num_rows),
        'avg_rating_180d': np.random.uniform(1, 5, size=num_rows),
        'category_diversity_180d': np.random.randint(1, 5, size=num_rows),
        'ticket_count_90d': np.random.randint(0, 5, size=num_rows),
        'negative_ticket_rate_90d': np.random.uniform(0, 1, size=num_rows),
        'avg_resolution_hours_90d': np.random.uniform(1, 72, size=num_rows),
        'days_since_signup': np.random.randint(30, 600, size=num_rows),
        'sessions_30d': np.random.randint(0, 30, size=num_rows),
        'product_views_30d': np.random.randint(0, 100, size=num_rows),
        'cart_adds_30d': np.random.randint(0, 10, size=num_rows),
        'wishlist_adds_30d': np.random.randint(0, 5, size=num_rows),
        'abandoned_carts_30d': np.random.randint(0, 5, size=num_rows),
        'email_opens_30d': np.random.randint(0, 20, size=num_rows),
        'campaign_clicks_30d': np.random.randint(0, 10, size=num_rows),
        'last_visit_days_ago': np.random.randint(0, 30, size=num_rows),
        # Target based roughly on recency and support to simulate real model patterns
        'churn_next_60d': np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3])
    })

if __name__ == "__main__":
    print("Generating synthetic data for API model deployment...")
    df = generate_training_data()
    X = df.drop(columns=['churn_next_60d'])
    y = df['churn_next_60d']

    num_cols = X.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='None')), 
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))
    ])

    print("Training model...")
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, 'model.pkl')
    print("Model successfully saved to 'model.pkl'. API is ready to run.")