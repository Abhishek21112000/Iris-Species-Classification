import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Standardize the data
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_standardized['species'] = df['species']

# Visualize the standardized data
sns.pairplot(df_standardized, hue="species", markers=["o", "s", "D"])
plt.suptitle('Pairwise scatter plots of the standardized Iris features', verticalalignment='bottom')
plt.show()

# K-means implementation from scratch
class KMeansScratch:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def inertia_(self, X):
        distances = np.linalg.norm(X - self.centroids[self.labels], axis=1)
        return np.sum(distances**2)

#K-means from scratch
kmeans_scratch = KMeansScratch(n_clusters=3)
X = df_standardized.iloc[:, :-1].values
kmeans_scratch.fit(X)
df_standardized['cluster_scratch'] = kmeans_scratch.labels

# Elbow KMeansScratch
def plot_elbow_curve_scratch(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans_scratch = KMeansScratch(n_clusters=k)
        kmeans_scratch.fit(X)
        distortions.append(kmeans_scratch.inertia_(X))

    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k (Scratch Implementation)')
    plt.show()

plot_elbow_curve_scratch(X)
"""
# K-means using scikit-learn
kmeans_sklearn = KMeans(n_clusters=3, random_state=42)
df_standardized['cluster_sklearn'] = kmeans_sklearn.fit_predict(X)

# Elbow KMeans (scikit-learn)
def plot_elbow_curve_sklearn(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k (scikit-learn)')
    plt.show()


plot_elbow_curve_sklearn(X)
"""
# Comparing with actual data
labels_actual = df_standardized['species'].cat.codes

# Confusion matrices
conf_matrix_scratch = confusion_matrix(labels_actual, df_standardized['cluster_scratch'])
#conf_matrix_sklearn = confusion_matrix(labels_actual, df_standardized['cluster_sklearn'])

# Scratch
ConfusionMatrixDisplay(conf_matrix_scratch, display_labels=iris.target_names).plot(cmap='Reds')
plt.title("Confusion Matrix for K-means (Scratch Implementation)")
plt.show()

"""#scikit-learn
ConfusionMatrixDisplay(conf_matrix_sklearn, display_labels=iris.target_names).plot(cmap='Blues')
plt.title("Confusion Matrix for K-means (scikit-learn)")
plt.show()
"""
# Accuracies
accuracy_scratch = accuracy_score(labels_actual, df_standardized['cluster_scratch'])
#accuracy_sklearn = accuracy_score(labels_actual, df_standardized['cluster_sklearn'])

print(f"Accuracy (Scratch Implementation): {accuracy_scratch:.4f}")
#print(f"Accuracy (scikit-learn): {accuracy_sklearn:.4f}")

# F1 Scores
def compute_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

f1_scratch = compute_f1_score(labels_actual, df_standardized['cluster_scratch'])
#f1_sklearn = compute_f1_score(labels_actual, df_standardized['cluster_sklearn'])

print(f'F1 Score (Scratch Implementation): {f1_scratch:.4f}')
#print(f'F1 Score (scikit-learn): {f1_sklearn:.4f}')
