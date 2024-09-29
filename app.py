import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('porto.ipynb')

# Title of the app
st.title("Fish Species Classification App")

# Instructions
st.write("Enter the features of the fish to classify its species.")

# Input fields for the features
length = st.number_input('Length (in cm)', min_value=0.0)
weight = st.number_input('Weight (in grams)', min_value=0.0)
width = st.number_input('Width (in cm)', min_value=0.0)
height = st.number_input('Height (in cm)', min_value=0.0)

# Prediction button
if st.button('Predict'):
    # Prepare the data for prediction
    input_data = np.array([[length, weight, width, height]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the result
    st.write(f"The predicted species is: {prediction[0]}")