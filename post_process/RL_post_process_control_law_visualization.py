# RL_post_process_control_law_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
#import networkx as nx


def cluster_control_analysis(X, U, n_clusters=10, random_state=42):
    # Step 1: KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Step 2: Transition Matrix P
    P = np.zeros((n_clusters, n_clusters))
    for i in range(1, len(labels)):
        P[labels[i - 1], labels[i]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)

    # Step 3: Average Actuation per Cluster
    avg_u = np.zeros((n_clusters, U.shape[1] if U.ndim > 1 else 1))
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        avg_u[i] = np.mean(U[cluster_indices], axis=0)

    # Step 4: MDS Projection
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=random_state)
    centroids_2d = mds.fit_transform(centroids)

    # Step 5: Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1], c=np.linalg.norm(avg_u, axis=1), cmap='viridis', s=300, edgecolors='k'
    )
    for i in range(n_clusters):
        ax.text(centroids_2d[i, 0], centroids_2d[i, 1], str(i), fontsize=12, ha='center', va='center', color='white')

    # Add arrows for transitions
    for i in range(n_clusters):
        for j in range(n_clusters):
            if P[i, j] > 0.05:  # only plot significant transitions
                ax.annotate(
                    '',
                    xy=centroids_2d[j],
                    xytext=centroids_2d[i],
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=P[i, j])
                )

    cbar = fig.colorbar(scatter, ax=ax, label='Avg Actuation Norm per Cluster')
    ax.set_title('Cluster-Based Control Law Visualization')
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    plt.tight_layout()
    plt.savefig("temp_fig.png", dpi=600)

    return labels, centroids, P, avg_u, centroids_2d


# Example usage (dummy data)
if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.rand(1000, 5)  # 5 sensors
    U = np.random.rand(1000, 1)  # 1 actuator
    cluster_control_analysis(X, U, n_clusters=10)
