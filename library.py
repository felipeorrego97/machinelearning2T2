import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for i in range(self.max_iters):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break
                
            self.centroids = new_centroids
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    

class KMedoids:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.medoids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # Asignar cada punto al medoide más cercano
            labels = self._assign_labels(X)
            
            # Calcular nuevos medoides
            new_medoids = self._find_medoids(X, labels)
            
            # Comprobar si los medoides han convergido
            if np.all(self.medoids == new_medoids):
                break
            
            self.medoids = new_medoids

    def predict(self, X):
        return self._assign_labels(X)

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(distances, axis=1)

    def _find_medoids(self, X, labels):
        new_medoids = np.empty_like(self.medoids)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            cluster_distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
            total_distances = np.sum(cluster_distances, axis=1)
            medoid_index = np.argmin(total_distances)
            new_medoids[i] = cluster_points[medoid_index]
        return new_medoids

