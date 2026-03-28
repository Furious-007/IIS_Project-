import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import kagglehub

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
# Download latest version
path = kagglehub.dataset_download("amisha0528/mall-customers-dataset")
print("Path to dataset files:", path)

data = pd.read_csv(os.path.join(path, "Mall_Customers.csv"))

print("Dataset Loaded Successfully!")
print(data.head())

# -----------------------------
# STEP 2: PREPROCESSING (TRAINING PREP)
# -----------------------------
# Select features
X = data.iloc[:, [3, 4]].values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData Preprocessed (Scaled)")

# -----------------------------
# STEP 3: MODEL TRAINING (K-MEANS)
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42)

# Training happens here 👇
kmeans.fit(X_scaled)

# Predict clusters
labels = kmeans.predict(X_scaled)

print("\nModel Training Completed!")

# -----------------------------
# STEP 4: EVALUATION
# -----------------------------
score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", round(score, 3))

# -----------------------------
# STEP 5: VISUALIZATION
# -----------------------------
plt.figure(figsize=(8,6))

for i in range(5):
    plt.scatter(
        X_scaled[labels == i, 0],
        X_scaled[labels == i, 1],
        label=f"Cluster {i}"
    )

# Centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("Customer Segmentation (K-Means)")
plt.xlabel("Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# STEP 6: HIERARCHICAL (OPTIONAL)
# -----------------------------
plt.figure(figsize=(10,6))
sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title("Dendrogram")
plt.show()

hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_scaled)

# -----------------------------
# STEP 7: SAVE OUTPUT
# -----------------------------
data['Cluster'] = labels
data.to_csv("Mall_Customers_Final.csv", index=False)

print("\nFinal clustered dataset saved!")