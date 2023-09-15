import numpy as np
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids


# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============


n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropically distributed data

random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
# blobs with varied variances
varied = datasets.make_blobs(
n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# Plot each dataset in separate figures
datasets_list = [blobs, no_structure, aniso, varied]
dataset_names = ["Blobs", "No Structure", "Anisotropic", "Varied Variances"]

# Define clustering algorithms
clustering_algorithms = {
    'K-Means': cluster.KMeans(n_clusters=3),
    'K-Medoids': KMedoids(n_clusters=3), 
    'DBSCAN': cluster.DBSCAN(eps=0.2, min_samples=10),
    'Spectral Clustering': cluster.SpectralClustering(n_clusters=3, gamma=5)
}

#eigen_solver='arpack'
# Iterate over datasets
for dataset, dataset_name in zip(datasets_list, dataset_names):
    X, y = dataset
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    plt.title(dataset_name)

    # Iterate over clustering algorithms
    for i, (name, algorithm) in enumerate(clustering_algorithms.items()):
        # Predict cluster labels
        if name == 'K-Medoids':
            algorithm.fit(X)
            labels = algorithm.predict(X)
        else:
            labels = algorithm.fit_predict(X)

        # Plot the clustering result
        plt.subplot(2, 2, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.title(name + dataset_name)

plt.show()

# Plot each dataset in separate figures
datasets_list = [noisy_circles, noisy_moons]
dataset_names = ["Noisy Circles", "Noisy Moons"]

# Define clustering algorithms
clustering_algorithms = {
    'K-Means': cluster.KMeans(n_clusters=2),
    'K-Medoids': KMedoids(n_clusters=2), 
    'DBSCAN': cluster.DBSCAN(eps=0.2, min_samples=10),
    'Spectral Clustering': cluster.SpectralClustering(n_clusters=2, gamma=20)
}

#eigen_solver='arpack'
# Iterate over datasets
for dataset, dataset_name in zip(datasets_list, dataset_names):
    X, y = dataset
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    plt.title(dataset_name)

    # Iterate over clustering algorithms
    for i, (name, algorithm) in enumerate(clustering_algorithms.items()):
        # Predict cluster labels
        if name == 'K-Medoids':
            algorithm.fit(X)
            labels = algorithm.predict(X)
        else:
            labels = algorithm.fit_predict(X)

        # Plot the clustering result
        plt.subplot(2, 2, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.title(name + dataset_name)

plt.show()