"""
RUN THIS COMMAND IN SHELL TO RUN THIS FILE

python -m K_Means.visualization
"""

import matplotlib.pyplot as plt
import numpy as np

from kMeansClustering import KMeans

x1 = np.array(
    [
        [1, 2],
        [2, 1],
        [2, 3],
        [3, 2],

        [8, 8],
        [9, 7],
        [8, 9],
        [9, 9],

        [2, 9],
        [3, 8],
        [1, 8],
        [2, 7],
    ]
)

model = KMeans(k=3)
model.fit(x1)

colors = ["red", "blue", "green", "orange", "purple", "brown"]

# plt.figure(figsize=(8, 6))

# Plot each cluster
for i, cluster in enumerate(model.clusters):
    cluster = np.array(cluster)

    plt.scatter(
        cluster[:, 0],
        cluster[:, 1],
        color=colors[i],
        s=35,
        label=f"Cluster {i}"
    )

# Plot centroids
centroids = np.array(model.centroids)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="black",
    marker="X",
    s=50,
    label="Centroids"
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()

plt.show()