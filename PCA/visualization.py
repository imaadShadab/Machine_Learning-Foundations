"""
RUN THIS COMMAND IN SHELL

python -m Principal_Component_Analysis.visualization
"""

import matplotlib.pyplot as plt
import numpy as np

from pca import PCA


x1 = np.array([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0],
    [5.0, 4.0],
    [6.0, 5.0],
    [7.0, 6.0],
    [8.0, 7.0],
    [9.0, 8.0],
])

# -----------------------------

pca = PCA(k=1)
pca.fit(x1)

x_transformed = pca.transform(x1)

# -----------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original Data
axes[0].scatter(
    x1[:, 0],
    x1[:, 1],
    color="royalblue",
    s=60
)

axes[0].set_title("Original Dataset")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].grid(True)

# Transformed Data
axes[1].scatter(
    x_transformed[:, 0],
    np.zeros(len(x_transformed)),
    color="crimson",
    s=60
)

axes[1].set_title("Dataset After PCA (k = 1)")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_yticks([])
axes[1].grid(True)

plt.tight_layout()
plt.show()