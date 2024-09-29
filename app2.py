import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Title of the web app
st.title("Fish Data Machine Learning Analysis")

# 1. Upload Dataset
st.header("1. Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display the dataset
    st.subheader("Dataset Preview")
    st.dataframe(df)

    # 2. Clean the Dataset
    st.header("2. Data Cleaning")

    # Handle missing values
    if st.checkbox("Show missing data summary"):
        st.write(df.isnull().sum())
        
    st.write("Handling missing values by filling with the mean (numerical) or mode (categorical).")
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    
    # Show cleaned dataset
    st.subheader("Cleaned Dataset")
    st.dataframe(df)

    # 3. Data Preprocessing
    st.header("3. Data Preprocessing")
    
    # Assuming the 'Species' column is the target column
    target_column = st.selectbox("Select the target column (label)", df.columns)
    
    # Encode categorical columns if necessary
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Split data into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Model Training
    st.header("4. Model Training")

    # Select the model
    model_choice = st.selectbox("Select a model", ["Logistic Regression", "K-Nearest Neighbors"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    else:
        model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Results
    st.header("5. Results")
    st.write(f"Model: {model_choice}")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Optionally display the predictions
    if st.checkbox("Show predictions"):
        st.write(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
