import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Property Recommender System", page_icon="🎯", layout="wide")

@st.cache_data
def get_recommendation_dataset():
    paths = ['data/cleaned.csv', 'cleaned.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

df = get_recommendation_dataset()

st.title("🎯 Tailored Property Recommendation Engine")
st.write("Find matching properties based on Location Advantages, Facilities, and Attribute Similarity.")

if not df.empty:
    rec_type = st.radio(
        "Select Recommendation Strategy",
        ["1. Property Attribute Similarity", "2. Location & Neighborhood Advantage", "3. Budget & Facility Matching"],
        horizontal=True
    )

    if rec_type == "1. Property Attribute Similarity":
        st.subheader("Find Similar Properties to a Target Listing")
        selected_address = st.selectbox("Select Target Property Address", options=df['address'].unique())
        
        target_idx = df[df['address'] == selected_address].index[0]
        target_row = df.loc[target_idx]
        
        st.write("**Target Spec:**", f"{target_row['property_type']} in {target_row['neighborhood']} | ${target_row['price']:,.0f} | {target_row['sqft']} sqft | {target_row['bedrooms']} bed, {target_row['bathrooms']} bath")

        # Feature matrix for similarity
        num_cols = ['sqft', 'land_sqft', 'bedrooms', 'bathrooms', 'price', 'property_age']
        scaler = StandardScaler()
        norm_features = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        
        sim_matrix = cosine_similarity(norm_features)
        sim_scores = list(enumerate(sim_matrix[target_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

        rec_indices = [i[0] for i in sim_scores]
        recs = df.iloc[rec_indices].copy()
        recs['Similarity Score'] = [f"{i[1]*100:.1f}%" for i in sim_scores]

        st.markdown("#### Top 5 Recommended Properties")
        st.dataframe(
            recs[['address', 'neighborhood', 'property_type', 'price', 'sqft', 'bedrooms', 'bathrooms', 'Similarity Score']],
            use_container_width=True
        )

    elif rec_type == "2. Location & Neighborhood Advantage":
        st.subheader("Location-Based Recommendation")
        target_neigh = st.selectbox("Preferred Neighborhood", options=df['neighborhood'].unique())
        max_dist_metro = st.slider("Maximum Metro Distance (Meters)", 100, 2000, 800)
        
        filtered_recs = df[
            (df['neighborhood'] == target_neigh) &
            (df['nearest_metro_m'] <= max_dist_metro)
        ].sort_values(by='price').head(5)
        
        if not filtered_recs.empty:
            st.dataframe(filtered_recs[['address', 'property_type', 'price', 'sqft', 'bedrooms', 'bathrooms', 'nearest_metro_m']], use_container_width=True)
        else:
            st.info("No properties found matching the exact metro distance criteria in this neighborhood.")

    elif rec_type == "3. Budget & Facility Matching":
        st.subheader("Facility & Budget Preference Matcher")
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Max Budget ($)", min_value=200000, max_value=10000000, value=1200000, step=50000)
            min_beds = st.slider("Min Bedrooms", 1, 6, 3)
        with col2:
            req_ac = st.checkbox("Require Central AC", value=True)
            req_fireplace = st.checkbox("Require Fireplace", value=False)
            
        matches = df[
            (df['price'] <= budget) &
            (df['bedrooms'] >= min_beds) &
            ((df['ac'] == 1.0) if req_ac else True) &
            ((df['fireplaces'] > 0) if req_fireplace else True)
        ].sort_values(by='price', ascending=False).head(5)
        
        st.dataframe(matches[['address', 'neighborhood', 'property_type', 'price', 'sqft', 'bedrooms', 'bathrooms', 'condition']], use_container_width=True)