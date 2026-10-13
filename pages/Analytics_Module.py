import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Market Analytics", page_icon="📊", layout="wide")

@st.cache_data
def get_data():
    paths = ['data/cleaned.csv', 'cleaned.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

df = get_data()

st.title("📊 Real Estate Exploratory Data Analytics")
st.write("Examine spatial distribution, price trends, feature correlations, and amenity insights.")

if not df.empty:
    # Sidebar Filters
    st.sidebar.header("Filter Analytics")
    selected_types = st.sidebar.multiselect(
        "Property Type", 
        options=df['property_type'].unique(),
        default=df['property_type'].unique()
    )
    
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=float(df['price'].min()),
        max_value=float(df['price'].max()),
        value=(float(df['price'].min()), float(df['price'].quantile(0.95)))
    )
    
    filtered_df = df[
        (df['property_type'].isin(selected_types)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ].copy() # Using copy to prevent SettingWithCopyWarning in tab3

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Spatial Map", "📈 Price Distribution", "📐 Attributes vs Price", "🏪 Amenities & Facilities"])

    with tab1:
        st.subheader("Geographical Distribution of D.C. Properties")
        
        # CHANGED: px.scatter_mapbox -> px.scatter_map
        fig_map = px.scatter_map(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="price",
            size="sqft",
            hover_name="address",
            hover_data=["neighborhood", "property_type", "bedrooms", "bathrooms"],
            color_continuous_scale="Viridis",
            zoom=11,
            height=600,
        )
        # CHANGED: mapbox_style -> map_style configured via layout updates
        fig_map.update_layout(map_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.subheader("Price Distribution & Boxplots")
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                filtered_df, x="price", color="property_type",
                marginal="rug", title="Price Distribution by Property Type",
                nbins=40, color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            fig_box = px.box(
                filtered_df, x="neighborhood", y="price",
                title="Price Variance Across Top Neighborhoods",
                color="neighborhood"
            )
            fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        st.subheader("Property Dimensions vs. Valuation")
        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                filtered_df, x="sqft", y="price", color="condition",
                hover_data=["address", "neighborhood"],
                trendline="ols",
                title="Living Area (SqFt) vs Price"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            filtered_df['price_per_sqft'] = filtered_df['price'] / filtered_df['sqft']
            fig_age = px.scatter(
                filtered_df, x="property_age", y="price_per_sqft",
                color="property_type", opacity=0.7,
                title="Property Age vs Price per SqFt"
            )
            st.plotly_chart(fig_age, use_container_width=True)

    with tab4:
        st.subheader("Nearby Amenities & Infrastructure Insights")
        if 'nearby_categories' in filtered_df.columns:
            # Flatten nearby category listings
            categories = filtered_df['nearby_categories'].dropna().apply(lambda x: eval(x) if isinstance(x, str) else x)
            cat_counts = pd.Series([cat for sublist in categories for cat in sublist]).value_counts()
            
            fig_cat = px.bar(
                x=cat_counts.index, y=cat_counts.values,
                labels={'x': 'Amenity Category', 'y': 'Count'},
                title="Frequency of Nearby Amenities",
                color=cat_counts.index
            )
            st.plotly_chart(fig_cat, use_container_width=True)
            
        if 'nearest_metro_m' in filtered_df.columns:
            fig_metro = px.box(
                filtered_df, x="property_type", y="nearest_metro_m",
                title="Distance to Nearest Metro Station (Meters)"
            )
            st.plotly_chart(fig_metro, use_container_width=True)
else:
    st.error("No data found! Please ensure 'cleaned.csv' exists in the base directory or 'data/' folder.")
