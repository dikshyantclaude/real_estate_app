import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Price Prediction Engine", page_icon="💰", layout="wide")

st.title("💰 Machine Learning Property Valuation Pipeline")
st.markdown("Estimate property prices using our trained Gradient Boosting ensemble regression model.")

@st.cache_resource
def load_model_artifacts():
    model_path = 'models/price_predictor_pipeline.pkl'
    schema_path = 'models/feature_names.json'
    
    if not os.path.exists(model_path):
        # Auto-train if artifacts are missing
        from train_model import train_and_save_model
        train_and_save_model()
        
    model = joblib.load(model_path)
    with open(schema_path) as f:
        schema = json.load(f)
    return model, schema

try:
    model, schema = load_model_artifacts()
    
    st.subheader("Property Specification Form")
    col1, col2, col3 = st.columns(3)

    with col1:
        sqft = st.number_input("Living Area (SqFt)", min_value=300, max_value=15000, value=1800, step=50)
        land_sqft = st.number_input("Land Area (SqFt)", min_value=200, max_value=70000, value=2500, step=100)
        bedrooms = st.slider("Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
        bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=10.0, value=2.5, step=0.5)

    with col2:
        stories = st.selectbox("Stories / Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0], index=2)
        property_age = st.number_input("Property Age (Years)", min_value=0, max_value=250, value=60)
        fireplaces = st.selectbox("Fireplaces", options=[0.0, 1.0, 2.0, 3.0, 4.0], index=1)
        ac = st.radio("Central Air Conditioning (AC)", options=[1.0, 0.0], format_func=lambda x: "Yes" if x == 1.0 else "No")

    with col3:
        neighborhood = st.selectbox("Neighborhood", options=schema['neighborhoods'], index=schema['neighborhoods'].index('Old City 2') if 'Old City 2' in schema['neighborhoods'] else 0)
        property_type = st.selectbox("Property Type", options=schema['property_types'])
        condition = st.selectbox("Condition Quality", options=schema['conditions'], index=schema['conditions'].index('Good') if 'Good' in schema['conditions'] else 0)
        
        # Approximate lat/lon defaults for DC
        latitude = st.number_input("Latitude", value=38.9158, format="%.4f")
        longitude = st.number_input("Longitude", value=-77.0360, format="%.4f")

    if st.button("🚀 Estimate Property Value", type="primary"):
        input_data = pd.DataFrame([{
            'sqft': sqft,
            'land_sqft': land_sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_age': property_age,
            'stories': stories,
            'ac': ac,
            'fireplaces': fireplaces,
            'latitude': latitude,
            'longitude': longitude,
            'property_type': property_type,
            'neighborhood': neighborhood,
            'condition': condition
        }])

        prediction = model.predict(input_data)[0]
        
        st.success(f"### Estimated Valuation: **${prediction:,.2f}**")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Estimated Price / SqFt:** ${prediction / sqft:,.2f}")
        with c2:
            st.info(f"**Valuation Range (±5%):** ${prediction * 0.95:,.2f} - ${prediction * 1.05:,.2f}")

except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    