import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('random_forest_model.pkl')

st.title("🌾 Crop Production Prediction App")

st.write("Predict crop production (in tons) using area, yield, and year.")

area = st.number_input("Area harvested", min_value=1.0, step=100.0)
yield_ = st.number_input("Yield", min_value=1.0, step=100.0)
year = st.slider("Year", min_value=2019, max_value=2023, value=2021)

if st.button("Predict Production"):
    input_data = np.array([[area, yield_, year]])
    prediction = model.predict(input_data)[0]
    st.success(f"🌾 Predicted Production: {prediction:,.2f} tons")

