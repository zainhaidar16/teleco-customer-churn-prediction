import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

import matplotlib.pyplot as plt

# Function to load data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to clean data
def clean_data(df):
    # Drop the customerID column
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    
    # Convert Gender to 0 and 1
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
    
    # Convert columns with 'Yes'/'No' to 1/0
    yes_no_columns = df.columns[df.isin(['Yes', 'No']).any()]
    df[yes_no_columns] = df[yes_no_columns].applymap(lambda x: 1 if x == 'Yes' else 0)
    
    # Handle missing values
    for column in df.columns:
        if df[column].dtype == 'object':  # Categorical column
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:  # Numeric column
            df[column].fillna(df[column].median(), inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

# Function to preprocess data
def preprocess_data(df, target_column):
    # Convert other categorical columns to numeric (one-hot encoding)
    df = pd.get_dummies(df, drop_first=True)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert non-numeric values to numeric
    if y.dtype == 'object':
        y = y.map({'No': 0, 'Yes': 1})
    
    return X, y

# Function to perform EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")

    # Display distribution of target variable
    st.write("Target Variable Distribution:")
    st.write(df['Churn'].value_counts())
    

    # Categorical feature distributions
    st.write("Categorical Feature Distributions:")
    for col in df.select_dtypes(include=['object']).columns:
        st.write(f"Distribution of {col}:")
        st.bar_chart(df[col].value_counts())
    

# Function to build and train the model
def build_and_train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    
    return model, history

# Streamlit app layout
def main():
    st.title("Customer Churn Prediction")

    st.write("This is a simple customer churn prediction app. Please upload your data and configure the settings to train the model.")

    # Sidebar for file upload and settings
    st.sidebar.header("Upload Data and Settings")

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.subheader("Data Overview")
        st.write(df.head())
        st.write("Data Types:")
        st.write(df.dtypes)

        # Checkbox for data cleaning
        if st.sidebar.checkbox("Clean Data"):
            df = clean_data(df)
            st.subheader("Cleaned Data")
            st.write(df.head())

        # Checkbox for EDA
        if st.sidebar.checkbox("Perform EDA"):
            perform_eda(df)
        
        # Checkbox for preprocessing
        if st.sidebar.checkbox("Preprocess Data"):
            target_column = st.sidebar.selectbox("Select Target Column", df.columns)
            X, y = preprocess_data(df, target_column)
            st.subheader("Preprocessed Data")
            st.write(X.head())
            st.write("Target Distribution")
            st.write(y.value_counts())
            
            # Checkbox to train model
            if st.sidebar.checkbox("Train Model"):
                _, history = build_and_train_model(X, y)
                st.subheader("Model Training Results")
                st.write(f"Final Accuracy: {history.history['accuracy'][-1]}")
                
                # Plot training history
                st.subheader("Training History")
                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'], label='Accuracy')
                ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
