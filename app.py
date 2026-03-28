import streamlit as st

st.set_page_config(
    page_title="Machine Learning Models",
    page_icon="🤖",
    layout="wide"
)

st.write("# Welcome to the Machine Learning Hub! 🤖")

st.markdown(
    """
    This is a unified web application showcasing three different machine learning models that you have built.
    
    ### Select a model from the sidebar to see it in action!
    
    ### Available Models
    1. ** Fashion Classification**
       - An image classification model built using **EfficientNetB0**.
       - Trains dynamically on the "Fashion Product Images" dataset.
       
    2. ** Air Quality Prediction**
       - A supervised learning model predicting Air Quality Index (AQI) in India.
       - Built using **Random Forest Regressor**.
       
    3. ** Customer Segmentation**
       - An unsupervised learning model clustering mall customers based on income and spending score.
       - Built using **K-Means Clustering** and **Hierarchical Clustering**.
    """
)
