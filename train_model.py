import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save_model():
    data_path = 'data/cleaned.csv'
    if not os.path.exists(data_path):
        data_path = 'cleaned.csv' # Fallback for flat layout
        
    df = pd.read_csv(data_path)

    # Feature definitions
    num_cols = [
        'sqft', 'land_sqft', 'bedrooms', 'bathrooms', 'property_age',
        'stories', 'ac', 'fireplaces', 'latitude', 'longitude'
    ]
    cat_cols = ['property_type', 'neighborhood', 'condition']
    
    X = df[num_cols + cat_cols]
    y = df['price']

    # Preprocessing Pipeline
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Final Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        ))
    ])

    print("Fitting Gradient Boosting Pipeline on DC Real Estate dataset...")
    model_pipeline.fit(X, y)

    # Evaluate on Dataset
    train_preds = model_pipeline.predict(X)
    print(f"Dataset Training MAE: ${mean_absolute_error(y, train_preds):,.2f}")
    print(f"Dataset Training R² Score: {r2_score(y, train_preds):.4f}")

    # Validate against inference_checks.json if available
    check_path = 'data/inference_checks.json' if os.path.exists('data/inference_checks.json') else 'inference_checks.json'
    if os.path.exists(check_path):
        with open(check_path) as f:
            checks = json.load(f)
        test_df = pd.DataFrame([c['inputs'] for c in checks])
        expected_y = np.array([c['expected'] for c in checks])
        
        check_preds = model_pipeline.predict(test_df)
        check_mae = mean_absolute_error(expected_y, check_preds)
        print(f"Inference Benchmark MAE: ${check_mae:,.2f}")

    # Serialize artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/price_predictor_pipeline.pkl')
    
    schema = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'neighborhoods': sorted(df['neighborhood'].unique().tolist()),
        'property_types': sorted(df['property_type'].unique().tolist()),
        'conditions': sorted(df['condition'].unique().tolist())
    }
    with open('models/feature_names.json', 'w') as f:
        json.dump(schema, f, indent=2)
        
    print("Model pipeline and schema successfully serialized to models/")

if __name__ == '__main__':
    train_and_save_model()