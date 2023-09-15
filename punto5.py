import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from library import KMedoids
from library import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Specifying the number of cluster our data should have
n_components = 4
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

X, y = make_blobs(
n_samples=500,
n_features=2,
centers=4,
cluster_std=1,
center_box=(-10.0, 10.0),
shuffle=True,
random_state=1,
)

plt.title("Unclustered Data")
plt.scatter(X[:, 0], X[:, 1], s=15)
plt.xticks([])
plt.yticks([])
plt.show()


for i in range(4):
    kmeans = KMeans(n_clusters=i+2)
    kmedoids = KMedoids(n_clusters=i+2)

    kmeans.fit(X)
    predicted_labels = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
    plt.legend()
    plt.title(f'KMean Clustering for {i+2}')
    plt.show()
    silhouette_avg = silhouette_score(X,predicted_labels)
    print(f'the silhouette average for {i+2} clusters in Kmeans is {silhouette_avg}')

    kmedoids.fit(X)
    predicted_labels = kmedoids.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
    plt.legend()
    plt.title(f'KMedoids Clustering for {i+2}')
    plt.show()
    silhouette_avg = silhouette_score(X,predicted_labels)
    print(f'the silhouette average for {i+2} clusters in kmedoids is {silhouette_avg}')
