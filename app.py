import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define custom orders for ordinal encoding
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])

st.title("💎 Diamond Price Predictor")

# Input fields
carat = st.number_input("Carat", min_value=0.0, step=0.01)
x = st.number_input("Length (x in mm)", min_value=0.0, step=0.01)
y = st.number_input("Width (y in mm)", min_value=0.0, step=0.01)
z = st.number_input("Depth (z in mm)", min_value=0.0, step=0.01)

cut = st.selectbox("Cut", cut_order)
color = st.selectbox("Color", color_order)
clarity = st.selectbox("Clarity", clarity_order)

if st.button("Predict"):
    # Calculate volume
    volume = x * y * z

    # Encode categorical features
    cat_input = [[cut, color, clarity]]
    encoded_cat = encoder.transform(cat_input)

    
    # Combine with numeric features
    input_features = np.array([[carat, encoded_cat[0][0], encoded_cat[0][1], encoded_cat[0][2], volume]])

    # Standardize features
    scaled_input = scaler.transform(input_features)

    # Predict
    prediction = model.predict(scaled_input)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")









