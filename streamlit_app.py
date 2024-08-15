import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Telco-Customer-Churn.csv')
    return data

# Clean the data
def clean_data(df):
    df_cleaned = df.dropna()
    return df_cleaned

# Preprocess the data
def preprocess_data(df):
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

# Display data and basic statistics
def display_data(df):
    st.write("## Data Overview")
    st.dataframe(df.head())
    st.write("### Basic Statistics")
    st.write(df.describe())

# Show EDA Visualizations
def eda_visualization(df):
    st.write("## Exploratory Data Analysis")
    st.write("### Churn Distribution")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    st.pyplot(plt.gcf())

    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt.gcf())

# Train and evaluate the model
def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, y_test, y_pred

# Display evaluation metrics
def display_metrics(y_test, y_pred):
    st.write("## Model Evaluation")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")

# Main function to run the Streamlit app
def main():
    st.title("Churn Prediction App")
    st.write("This app predicts customer churn based on their details.")

    # Load Data
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
        st.session_state['cleaned'] = False
        st.session_state['preprocessed'] = False
        st.session_state['eda_done'] = False

    if not st.session_state['data_loaded']:
        st.session_state['data'] = load_data()
        display_data(st.session_state['data'])
        st.session_state['data_loaded'] = True

    # Clean Data
    if st.button("Clean Data"):
        st.session_state['data'] = clean_data(st.session_state['data'])
        st.session_state['cleaned'] = True
        st.session_state['preprocessed'] = False
        st.session_state['eda_done'] = False

    if st.session_state['cleaned']:
        st.write("### Cleaned Data")
        st.dataframe(st.session_state['data'].head())

        # Preprocess Data
        if st.button("Preprocess Data"):
            st.session_state['data'] = preprocess_data(st.session_state['data'])
            st.session_state['preprocessed'] = True

    if st.session_state['preprocessed']:
        st.write("### Preprocessed Data")
        st.dataframe(st.session_state['data'].head())

        # Show Statistics
        st.write("### Basic Statistics after Preprocessing")
        st.write(st.session_state['data'].describe())

        # EDA Visualizations
        if st.button("Show EDA Visualizations"):
            eda_visualization(st.session_state['data'])
            st.session_state['eda_done'] = True

    if st.session_state['eda_done']:
        # Train Model
        if st.button("Train Model"):
            model, y_test, y_pred = train_model(st.session_state['data'])
            display_metrics(y_test, y_pred)

if __name__ == "__main__":
    main()
