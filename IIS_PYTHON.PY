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

def main():
    print("Downloading dataset using kagglehub...")
    # Download latest version of the dataset
    path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
    print(f"Dataset downloaded to: {path}")

    # Inspect the downloaded files
    dataset_files = os.listdir(path)
    print(f"\nFiles found in dataset directory: {dataset_files}")

    # The dataset typically contains several files. We'll use 'city_day.csv' 
    # which has aggregate AQI per city over days.
    csv_path = os.path.join(path, "city_day.csv")
    
    if not os.path.exists(csv_path):
        # Fallback to finding any CSV file if 'city_day.csv' doesn't exist
        csv_files = [f for f in dataset_files if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("Could not find any CSV files in the downloaded dataset.")
        csv_path = os.path.join(path, csv_files[0])

    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Initial Dataset Shape: {df.shape}")

    # For supervised learning, let's predict the Air Quality Index (AQI).
    # We must drop rows where the target 'AQI' is missing.
    if 'AQI' not in df.columns:
        raise ValueError("Expected target column 'AQI' is not present in the dataset.")
        
    df = df.dropna(subset=['AQI'])
    print(f"Dataset Shape after dropping missing target values: {df.shape}")

    # ------------------ PREPROCESSING ------------------
    # Drop columns that are either non-numeric identifiers, leakage, or irrelevant.
    # - AQI_Bucket is a categorical class for the AQI target.
    # - Date is time. 
    # - City/StationId are identifiers.
    columns_to_drop = ['AQI', 'AQI_Bucket', 'Date', 'City', 'StationId']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    X = df.drop(columns=columns_to_drop)
    y = df['AQI']

    # One-hot encode object/categorical columns, and impute missing values out of necessity
    # for Random Forest which requires a fully populated numeric matrix.
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")

    # ------------------ 1. Random Forest Regressor ------------------
    print("\n--- Training Random Forest Regressor ---")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_r2 = r2_score(y_test, rf_predictions)
    print(f"[Random Forest] RMSE: {rf_rmse:.2f}")
    print(f"[Random Forest] R2 Score: {rf_r2:.4f}")

    # ------------------ 2. VISUALIZATIONS ------------------
    print("\nGenerating visualizations. Close the plot windows to finish execution...")
    sns.set_theme(style="whitegrid")

    # Visualization 1: Feature Importances
    plt.figure(figsize=(10, 6))
    rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    rf_importances.sort_values(ascending=True).tail(10).plot(kind='barh', color='teal')
    plt.title("Random Forest - Top 10 Feature Importances for predicting AQI")
    plt.xlabel("Importance Score")
    plt.ylabel("Pollutant/Feature")
    plt.tight_layout()
    
    # Visualization 2: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, rf_predictions, alpha=0.4, color='indigo')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title("Random Forest - Actual vs. Predicted AQI")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.tight_layout()

    # Visualization 3: Correlation Matrix
    plt.figure(figsize=(12, 10))
    # Select only numeric features for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap of Variables")
    plt.tight_layout()

    # Show all plots interactively
    plt.show()

if __name__ == "__main__":
    main()
