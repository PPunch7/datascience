# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
# %matplotlib inline

# Load the dataset
df = pd.read_csv('wk06/Mall_Customers.csv')

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Preprocessing
# Encode Gender feature
if df['Gender'].dtype == 'object':
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Select some features for clustering
x = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------------------------------------------------------------------
### Assignment: Exploratory Data Analysis

# Summary Statistics
df.describe(include='all')

# Distribution of Numerical Features
df[['Age','Annual Income (k$)','Spending Score (1-100)']].hist(
    figsize=(12,6), bins=15, edgecolor='black')
plt.show()

# Distribution of Categorical Feature (Gender)
sns.countplot(x=df['Gender'])
plt.show()

# Boxplots
sns.boxplot(data=df[['Age','Annual Income (k$)','Spending Score (1-100)']])
plt.show()
## The average client age is approximately 38 years old.
## Average annual income is approximately $60,000.
## Average spending score is ~50, with most clients scoring between 40-60.
## Gender distribution is nearly equal between male and female.
##

# Pairplot
sns.pairplot(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde')
plt.show()
## Income and Spending Scores show a pattern of groups.
## Age vs. Score indicates that customers aged 20-40 have higher spending habits.
##

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
## Most correlations are quite low.
## Age and Spending Score have a negative correlation (~-0.33) -> older people tend to spend less.
## Income is not strongly correlated with Spending Score,
## which means Spending Score may be more correlated with lifestyle than earning.
##

# Gender-based Spending Analysis
sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df)
plt.show()
## Females have a slightly higher average spending score than males.
## The difference isn't extremely significant.
##

# Incom vs Spending Score Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, hue='Gender')
plt.show()
## Clearly there are 4–5 clusters naturally:
##      Low income - low spending
##      Low income - high spending
##      High income - low spending
##      High income - high spending
##      Moderate income - Moderate spending

# ------------------------------------------------------------------------------------------
print("------------------------------------------------------------------------")

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(x_scaled)
df['KMeans_Cluster'] = kmeans.labels_

# Visualize K-Means Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Cluster', data=df, palette='viridis')
plt.title('K-Means Clustering (K=5)')
plt.show()

## Hierarchical Clustering
# Perform Agglomerative Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=5)
df['Agglo_Cluster'] = agglo.fit_predict(x_scaled)

# Visualize Hierarchical Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Agglo_Cluster', data=df, palette='viridis')
plt.title('Agglomerative Hierarchical Clustering (K=5)')
plt.show()

# Dendrogram for Hierarchical Clustering
linked = linkage(x_scaled, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.show()

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(x_scaled)

# Visualize DBSCAN Clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_Cluster', data=df, palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()

### Evaluate Clustering Performance ###
print("\nK-Means Silhouette Score:", silhouette_score(x_scaled, kmeans.labels_))
print("Agglomerative Silhouette Score:", silhouette_score(x_scaled, agglo.labels_))
print("DBSCAN Silhouette Score:", silhouette_score(x_scaled, dbscan.labels_))

print("\nK-Means Calinski-Harabasz Index:", calinski_harabasz_score(x_scaled, kmeans.labels_))
print("Agglomerative Calinski-Harabasz Index:", calinski_harabasz_score(x_scaled, agglo.labels_))
print("DBSCAN Calinski-Harabasz Index:", calinski_harabasz_score(x_scaled, dbscan.labels_))

print("\nK-Means Davies-Bouldin Index:", davies_bouldin_score(x_scaled, kmeans.labels_))
print("Agglomerative Davies-Bouldin Index:", davies_bouldin_score(x_scaled, agglo.labels_))
print("DBSCAN Davies-Bouldin Index:", davies_bouldin_score(x_scaled, dbscan.labels_))

print("------------------------------------------------------------------------")
# ------------------------------------------------------------------------------------------

### Assignment: 
## 1. Perform a visual comparison of these metrics and discuss.
# Visual Comparison of Clustering Metrics
metrics_df = pd.DataFrame({
    'Model': ['K-Means', 'Agglomerative', 'DBSCAN'],
    'Silhouette': [
        silhouette_score(x_scaled, kmeans.labels_),
        silhouette_score(x_scaled, agglo.labels_),
        silhouette_score(x_scaled, dbscan.labels_)
    ],
    'Calinski-Harabasz': [
        calinski_harabasz_score(x_scaled, kmeans.labels_),
        calinski_harabasz_score(x_scaled, agglo.labels_),
        calinski_harabasz_score(x_scaled, dbscan.labels_)
    ],
    'Davies-Bouldin': [
        davies_bouldin_score(x_scaled, kmeans.labels_),
        davies_bouldin_score(x_scaled, agglo.labels_),
        davies_bouldin_score(x_scaled, dbscan.labels_)
    ]
})

print("\n=== Clustering Metrics Summary ===")
print(metrics_df)

# Plot the metrics
metrics_df_plot = metrics_df.set_index('Model')

metrics_df_plot[['Silhouette','Calinski-Harabasz']].plot(kind='bar', figsize=(12,6))
plt.title("Cluster Quality Comparison: Silhouette & Calinski-Harabasz")
plt.ylabel("Score (Higher is Better)")
plt.show()

metrics_df_plot[['Davies-Bouldin']].plot(kind='bar', color='orange', figsize=(12,6))
plt.title("Cluster Quality Comparison: Davies-Bouldin")
plt.ylabel("Score (Lower is Better)")
plt.show()

## K-Means and Agglomerative Clustering significantly outperform DBSCAN on this dataset.
## The Silhouette Score and the Calinski–Harabasz Index, which both reward well-separated and compact clusters,
## show much higher values for K-Means and Hierarchical clustering.
## Conversely, DBSCAN receives a very low Silhouette Score and a high Davies–Bouldin Index,
## indicating poorly defined or irregular cluster structures.

## The Davies–Bouldin Index, where lower values represent better clustering performance,
## also favours K-Means and Agglomerative techniques. The results are consistent with the visual scatter plots:
## DBSCAN tends to assign many points to noise (-1 cluster) because the dataset 
## does not exhibit density-based clusters but rather separated geometric groups. 

## Overall, both K-Means and Hierarchical clustering perform strongly on the Mall Customers dataset, 
## while DBSCAN is not suitable due to the nature of the data distribution, which lacks 
## density-based grouping patterns.

# ------------------------------------------------------------------------------------------

## 2. Perform these clustering techniques on another dataset of your choice and present findings.
# Clustering on Another Dataset (Iris Dataset)
from sklearn.datasets import load_iris

# Load Data
iris = load_iris()
x_iris = iris.data
iris_df = pd.DataFrame(x_iris, columns=iris.feature_names)
print("\nIris Dataset Shape:", iris_df.shape)
print(iris_df.head())

# Scale
scaler = StandardScaler()
x_iris_scaled = scaler.fit_transform(iris_df)

# K-Means
kmeans_iris = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans_iris.fit_predict(x_iris_scaled)
iris_df['KMeans_Cluster'] = kmeans_labels

# Agglomerative
agglo_iris = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo_iris.fit_predict(x_iris_scaled)
iris_df['Agglo_Cluster'] = agglo_labels

# DBSCAN
dbscan_iris = DBSCAN(eps=0.4, min_samples=4)
dbscan_labels = dbscan_iris.fit_predict(x_iris_scaled)
iris_df['DBSCAN_Cluster'] = dbscan_labels

## Visualise
# K-Means
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='KMeans_Cluster', data=iris_df, palette='viridis')
plt.title('K-Means Clustering (K=3)')
plt.show()

# Agglomerative
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='Agglo_Cluster', data=iris_df, palette='viridis')
plt.title('Agglomerative Hierarchical Clustering (K=3)')
plt.show()

# DBSCAN
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='DBSCAN_Cluster', data=iris_df, palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()

#๒ Evaluate
iris_metrics = pd.DataFrame({
    'Model': ['K-Means','Agglomerative','DBSCAN'],
    'Silhouette': [
        silhouette_score(x_iris_scaled, kmeans_labels),
        silhouette_score(x_iris_scaled, agglo_labels),
        silhouette_score(x_iris_scaled, dbscan_labels)
    ],
    'Calinski-Harabasz': [
        calinski_harabasz_score(x_iris_scaled, kmeans_labels),
        calinski_harabasz_score(x_iris_scaled, agglo_labels),
        calinski_harabasz_score(x_iris_scaled, dbscan_labels)
    ],
    'Davies-Bouldin': [
        davies_bouldin_score(x_iris_scaled, kmeans_labels),
        davies_bouldin_score(x_iris_scaled, agglo_labels),
        davies_bouldin_score(x_iris_scaled, dbscan_labels)
    ]
})

print("\n=== Iris Dataset Clustering Metrics ===")
print(iris_metrics)

# Plot the metrics
metrics_df_plot = iris_metrics.set_index('Model')

metrics_df_plot[['Silhouette','Calinski-Harabasz']].plot(kind='bar', figsize=(12,6))
plt.title("Cluster Quality Comparison: Silhouette & Calinski-Harabasz")
plt.ylabel("Score (Higher is Better)")
plt.show()

metrics_df_plot[['Davies-Bouldin']].plot(kind='bar', color='orange', figsize=(12,6))
plt.title("Cluster Quality Comparison: Davies-Bouldin")
plt.ylabel("Score (Lower is Better)")
plt.show()

## For the Iris dataset, K-Means again provides the best clustering performance,
## achieving the highest Silhouette Score and Calinski–Harabasz Index.
## This is expected because the Iris dataset is well known for having three well-defined clusters 
## corresponding to the three species. 
## Agglomerative Clustering performs comparably well, whereas DBSCAN performs
## poorly due to the lack of density-based cluster structures in this dataset.

## Similar to the Mall Customers dataset, DBSCAN struggles when the dataset 
## has geometric clusters instead of density-separated patterns. 
## K-Means is particularly effective because the clusters in Iris are roughly convex 
## and evenly shaped, matching the assumptions of the K-Means algorithm.
