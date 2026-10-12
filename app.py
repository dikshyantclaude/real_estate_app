import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="DC Real Estate Analytics & AI Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_main_data():
    paths = ['data/cleaned.csv', 'cleaned.csv']
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            return df
    st.error("Data file 'cleaned.csv' not found.")
    return pd.DataFrame()

df = load_main_data()

st.title("🏛️ Washington D.C. Real Estate Intelligence Platform")
st.markdown("""
Welcome to the **Capstone Real Estate Data Science Application**. This platform delivers end-to-end spatial analytics, predictive price valuation, and multi-criteria property recommendations tailored for the D.C. housing market.
""")

st.divider()

if not df.empty:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Properties", f"{len(df):,}")
    m2.metric("Median Home Price", f"${df['price'].median():,.0f}")
    m3.metric("Avg. Price per SqFt", f"${(df['price'] / df['sqft']).mean():,.2f}")
    m4.metric("Avg. Property Age", f"{df['property_age'].mean():.0f} Yrs")
    m5.metric("Neighborhoods Covered", f"{df['neighborhood'].nunique()}")

    st.markdown("### Quick Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 **Analytics Module**\n\nInteractive spatial maps, price distribution analytics, boxplots, and nearby amenity density analyses.")
    with col2:
        st.success("💰 **Price Valuation**\n\nMachine learning valuation pipeline predicting estimated market price based on property characteristics.")
    with col3:
        st.warning("🎯 **Recommender System**\n\nTailored property recommendations based on location advantages, facilities, and price similarity.")