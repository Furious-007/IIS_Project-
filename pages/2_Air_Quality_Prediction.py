import streamlit as st
import os
import pandas as pd
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Air Quality Prediction", page_icon="🌍")

st.markdown("#  Air Quality Prediction")
st.write("This application downloads Indian air quality dataset, trains a Random Forest model on-the-fly, and predicts the Air Quality Index (AQI).")

@st.cache_data
def load_data():
    with st.spinner("Downloading dataset using kagglehub..."):
        path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
    
    dataset_files = os.listdir(path)
    csv_path = os.path.join(path, "city_day.csv")
    
    if not os.path.exists(csv_path):
        csv_files = [f for f in dataset_files if f.endswith('.csv')]
        if not csv_files:
            st.error("Could not find any CSV files in the downloaded dataset.")
            st.stop()
        csv_path = os.path.join(path, csv_files[0])

    df = pd.read_csv(csv_path)
    return df

df = load_data()

st.subheader("Data Overview")
st.write(f"Initial Dataset Shape: `{df.shape}`")
st.dataframe(df.head())

# Filter for AQI
if 'AQI' not in df.columns:
    st.error("Expected target column 'AQI' is not present in the dataset.")
    st.stop()

df = df.dropna(subset=['AQI'])
st.write(f"Dataset Shape after dropping missing target values: `{df.shape}`")

# ------------------ PREPROCESSING ------------------
columns_to_drop = ['AQI', 'AQI_Bucket', 'Date', 'City', 'StationId']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=columns_to_drop)
y = df['AQI']

# One-hot encode and impute
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("### Model Training")
st.write("Training **Random Forest Regressor** with 100 estimators...")

@st.cache_resource
def train_model(X_tr, y_tr):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    return model

with st.spinner("Training model..."):
    rf_model = train_model(X_train, y_train)

st.success("Model trained successfully!")

rf_predictions = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

col1, col2 = st.columns(2)
col1.metric("RMSE (Root Mean Squared Error)", f"{rf_rmse:.2f}")
col2.metric("R2 Score", f"{rf_r2:.4f}")

# ------------------ 2. VISUALIZATIONS ------------------
st.write("---")
st.write("###  Visualizations")
sns.set_theme(style="whitegrid")

# Visualization 1
st.subheader("Top 10 Feature Importances")
fig1, ax1 = plt.subplots(figsize=(10, 6))
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importances.sort_values(ascending=True).tail(10).plot(kind='barh', color='teal', ax=ax1)
ax1.set_title("Random Forest - Top 10 Feature Importances for predicting AQI")
ax1.set_xlabel("Importance Score")
ax1.set_ylabel("Pollutant/Feature")
st.pyplot(fig1)

# Visualization 2
st.subheader("Actual vs. Predicted AQI")
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(y_test, rf_predictions, alpha=0.4, color='indigo')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title("Random Forest - Actual vs. Predicted AQI")
ax2.set_xlabel("Actual AQI")
ax2.set_ylabel("Predicted AQI")
st.pyplot(fig2)

# Visualization 3
st.subheader("Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax3)
ax3.set_title("Correlation Heatmap of Variables")
st.pyplot(fig3)
