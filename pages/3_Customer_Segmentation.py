import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import kagglehub
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")

st.markdown("# Customer Segmentation")
st.write("Segmenting mall customers based on Annual Income and Spending Score using **K-Means Clustering**.")

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("amisha0528/mall-customers-dataset")
    data = pd.read_csv(os.path.join(path, "Mall_Customers.csv"))
    return data

with st.spinner("Loading Data..."):
    data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Preprocessing
X = data.iloc[:, [3, 4]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write("---")
st.write("### Model Training (K-Means)")
with st.spinner("Training K-Means Model..."):
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

st.success("Model Training Completed!")
st.metric("Silhouette Score", f"{score:.3f}")

st.write("###  K-Means Clusters")
fig, ax = plt.subplots(figsize=(8,6))
for i in range(5):
    ax.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f"Cluster {i}")

centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, label='Centroids', color='black')

ax.set_title("Customer Segmentation (K-Means)")
ax.set_xlabel("Annual Income (Scaled)")
ax.set_ylabel("Spending Score (Scaled)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.write("---")
st.write("### 🌳 Hierarchical Clustering (Dendrogram)")
st.write("We can also visualize the hierarchy of clusters using a dendrogram.")
fig_dend, ax_dend = plt.subplots(figsize=(10,6))
sch.dendrogram(sch.linkage(X_scaled, method='ward'), ax=ax_dend)
ax_dend.set_title("Dendrogram")
st.pyplot(fig_dend)

data['Cluster'] = labels
st.download_button(
    label="Download Clustered Dataset as CSV",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='Mall_Customers_Final.csv',
    mime='text/csv',
)
