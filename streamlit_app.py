import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Function to load data
def load_data(file):
    df = pd.read_csv(file)
    df.drop(["customerID"],axis="columns",inplace=True)
    return df

# Function to display data information
def data_overview(df):
    st.subheader("Data Overview")
    st.write("Shape of the dataset:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("First few rows of the dataset:")
    st.write(df.head())

# Function for data cleaning
def clean_data(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Handle missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df

# Function for feature engineering
def feature_engineering(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Only encode columns with less than 10 unique values
            df[col] = LabelEncoder().fit_transform(df[col])
    
    return df

# Function for data preprocessing
def preprocess_data(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Function for EDA
def eda(df):
    st.subheader("Exploratory Data Analysis")
    
    # Plot 1: Distribution of target variable
    if 'Churn' in df.columns:
        st.write("1. Distribution of Target Variable")
        sns.countplot(x='Churn', data=df)
        st.pyplot()
    
    # Plot 2: Correlation heatmap
    st.write("2. Correlation Heatmap")
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot()
    
    # Plot 3: Bar plots for categorical features
    st.write("3. Bar Plot for Categorical Features")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        sns.countplot(y=col, data=df)
        st.pyplot()
    
    # Plot 4: Box plots for numerical features
    st.write("4. Box Plot for Numerical Features")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        sns.boxplot(x=df[col])
        st.pyplot()

    # Plot 5: Pairplot of key features
    st.write("5. Pairplot")
    sns.pairplot(df)
    st.pyplot()

# Function to train the model
def train_model(df, model_choice):
    st.subheader("Model Training and Evaluation")
    
    if 'Churn' not in df.columns:
        st.error("Target column 'Churn' not found in the dataset.")
        return None
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_choice == 'Logistic Regression':
        model = LogisticRegression()
    elif model_choice == 'Random Forest':
        model = RandomForestClassifier()
    elif model_choice == 'Support Vector Machine':
        model = SVC(probability=True)
    elif model_choice == 'Gradient Boosting':
        model = GradientBoostingClassifier()
    elif model_choice == 'K-Nearest Neighbors':
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))
    
    return model

# Streamlit app layout
st.title("Customer Churn Prediction App")

# Sidebar for settings
st.sidebar.header("Settings")

# Data Collection: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    data_overview(df)
    
    # Checkbox options for various steps
    if st.sidebar.checkbox("Clean Data"):
        df = clean_data(df)
        st.write("Cleaned Data:")
        st.write(df.head())
    
    if st.sidebar.checkbox("Preprocess Data"):
        df = feature_engineering(df)
        df = preprocess_data(df)
        st.write("Preprocessed Data:")
        st.write(df.head())
    
    if st.sidebar.checkbox("Perform EDA"):
        eda(df)
    
    model_choice = st.sidebar.selectbox("Choose Model", 
                                        ["Logistic Regression", "Random Forest", "Support Vector Machine", 
                                         "Gradient Boosting", "K-Nearest Neighbors"])
    
    if st.sidebar.checkbox("Train Model"):
        model = train_model(df, model_choice)
