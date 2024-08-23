# 🎈 Customer Churn Prediction Application

This repository contains code for a simple customer churn prediction app using Streamlit. The app allows users to upload their own CSV data, clean the data, perform exploratory data analysis (EDA), preprocess the data, and train a model to predict customer churn.

### How to Run the App

To run the app on your own machine, follow these steps:

1. Install the required packages by running the following command in your terminal:

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app using the following command:

   ```
   $ streamlit run streamlit_app.py
   ```

### Functions

The `streamlit_app.py` file contains several functions that are used in the app:

- `load_data(uploaded_file)`: This function takes an uploaded CSV file as input and returns a pandas DataFrame containing the data.

- `clean_data(df)`: This function cleans the data by dropping the `customerID` column, converting the `gender` column to numeric values (0 for "Female" and 1 for "Male"), converting columns with "Yes" and "No" values to 1s and 0s, handling missing values by filling them with the mode for categorical columns and the median for numeric columns, and dropping duplicate rows. It returns the cleaned DataFrame.

- `preprocess_data(df, target_column)`: This function preprocesses the data by converting categorical columns to numeric values using one-hot encoding, separating the features and target variable, and converting the target variable to numeric values (0 for "No" and 1 for "Yes"). It returns the preprocessed features (`X`) and target variable (`y`).

- `perform_eda(df)`: This function performs exploratory data analysis (EDA) by displaying the distribution of the target variable (`Churn`) and the distribution of categorical features using bar charts.

- `build_and_train_model(X_train, y_train)`: This function builds and trains a simple neural network model using TensorFlow and Keras. The model architecture consists of three dense layers with ReLU activation functions and a sigmoid activation function in the output layer. It compiles the model with the Adam optimizer and binary cross-entropy loss, and trains the model for 10 epochs with a batch size of 32. It returns the trained model and the training history.

- `main()`: This function is the entry point of the Streamlit app. It sets up the app layout, including the title and description, file upload and settings sidebar, and the main content area. It calls the other functions based on the user's selections.

Feel free to explore the code and customize it for your own customer churn prediction project.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-predictions.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
