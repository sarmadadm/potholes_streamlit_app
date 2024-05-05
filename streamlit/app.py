import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


st.write("""
    # Welcome to NYC Pothole Prediction App!
    
    This app is demonstrating the 
    - Feature Distributions | Select from Sidebar
    - Key insights | Select from Sidebar
    - Predictions | Input your Features
    - Predictor Feature Importance | Plot to Display the Important Features
    
    Feel free to explore different sections using the sidebar.
    """)


df = pd.read_csv('Data/numeric_df.csv')



# -------------------------------------------------------------------------------
# distribution of every feature
st.sidebar.title("Select a Feature to see Distribution")
default_column = df.columns[4]  # Select the first column as default
selected_column = st.sidebar.selectbox('Select Feature', options=[''] + list(df.columns), index=0)
    
if selected_column:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[selected_column], bins=100, color='skyblue', edgecolor='black')
    plt.title('Histogram of {}'.format(selected_column))
    plt.xlabel(selected_column)
    plt.ylabel('Frequency')
    plt.grid(True)
    st.pyplot()

# -------------------------------------------------------------------------------
# load visuals
# Function to load and display image
def load_image(image_path):
    image = Image.open(image_path)
    st.image(image, caption='Selected Image', use_column_width=True)

# Define paths to the images
image_paths = {
    'Image 3': '../src/images/boro_potholes.png',
    'Image 2': '../src/images/monthly_potholes.png',
    'Image 1': '../src/images/zipcode_choropleth.png'
}

# Preload the images
preloaded_images = {key: Image.open(image_path) for key, image_path in image_paths.items()}

st.sidebar.title("Select a Visual for more Insights")

# Add buttons to select an image on the sidebar
selected_image = None
if st.sidebar.button("Choropleth Map of Zipcodes"):
    selected_image = 'Image 1'
elif st.sidebar.button("Monthly Number of Potholes"):
    selected_image = 'Image 2'
elif st.sidebar.button("Distribution of Potholes"):
    selected_image = 'Image 3'

# Display the selected image on the main page
if selected_image:
    load_image(image_paths[selected_image])

# ----------------------------------------------------------------------------------
# predictions

import joblib
import matplotlib.pyplot as plt

st.write("""
    ##### Please Enter the Following Inputs for the model to predict the Number of Days to Fix a Pothole
    """)

# Load the pickled Random Forest Regressor model
@st.cache_resource()
def load_model(model_path):
    return joblib.load(model_path)

# Load the first row of your dataset to get default features
def load_first_row(dataset_path):
    df = pd.read_csv(dataset_path)
    df.head()
    df = df.drop(['days_to_fix'], axis=1)
    
    return df.iloc[0]

# Function to make predictions
def predict(model, features):
    # Preprocess the features if needed (e.g., encoding, scaling)
    # Combine default features with user-input features
    default_features = load_first_row('../Data/num_df.csv')
    combined_features = {**default_features, **features}
    # Make predictions using the model
    prediction = model.predict(pd.DataFrame([combined_features]))
    # Get feature importances
    feature_importances = model.feature_importances_
    return prediction, feature_importances

# Load the model
model = load_model('../models/random_forest_model.joblib')

# Define user-input features
zip_code = st.number_input('Zip Code', value=10001, step=1)
month_num = st.number_input('Month Number', value=1, step=1)
boro_num = st.number_input('Borough Number', value=1, step=1)
tmin = st.number_input('Minimum Temperature', value=0.0, step=0.1)
shape_leng = st.number_input('Area of a Pothole', value=0.0, step=0.1)

# Prepare user-input features
user_features = {
    'zip_code': zip_code,
    'month_num': month_num,
    'Boro_num': boro_num,
    'TMIN': tmin,
    'Shape_Leng': shape_leng
}

# Make predictions
prediction, feature_importances = predict(model, user_features)

# Display the prediction
st.write("Predicted number of days to fix pothole:", prediction)

st.write('##### Following are the 10 most Important Predictors for your Input')
# Function to plot feature importances
def plot_feature_importances(feature_importances):
    # Get the names of the features
    feature_names = load_first_row('../Data/num_df.csv').index.values
    
    # Select the top 10 most important features
    top_indices = feature_importances.argsort()[-10:][::-1]
    top_features = feature_names[top_indices]
    top_importances = feature_importances[top_indices]
    
    # Plot the top 10 most important features
    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Most Important Features')
    st.pyplot()

# Plot the top 10 most important features
plot_feature_importances(feature_importances)