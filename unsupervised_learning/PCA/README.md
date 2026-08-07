# Principal Component Analysis (PCA) from Scratch

A NumPy implementation of **Principal Component Analysis (PCA)** built entirely from scratch to understand the mathematics behind dimensionality reduction.

Unlike supervised learning algorithms, PCA does **not** make predictions or require labeled data. Instead, it discovers a new coordinate system that preserves as much information (variance) as possible while reducing the number of features.

---

## Mathematical Intuition

Imagine a dataset with highly correlated features.

Instead of representing the data using the original feature axes, PCA finds **new orthogonal axes** (Principal Components) along which the data varies the most.

The first principal component captures the greatest amount of variance.

The second principal component captures the next highest variance while remaining perpendicular to the first.

By keeping only the top principal components, we reduce dimensionality while retaining most of the information contained in the dataset.

---

## PCA Pipeline

1. Center the dataset by subtracting the mean of every feature.
2. Compute the covariance matrix.
3. Compute the eigenvalues and eigenvectors of the covariance matrix.
4. Sort the eigenvalues in descending order.
5. Reorder the eigenvectors accordingly.
6. Select the top **k** eigenvectors.
7. Form the projection matrix.
8. Project the original data into the new feature space.

---

## Mathematical Formulas

### 1. Center the Data

\[
X_{centered}=X-\mu
\]

---

### 2. Covariance Matrix

\[
\Sigma=\frac{1}{m-1}X_{centered}^{T}X_{centered}
\]

---

### 3. Eigen Decomposition

\[
\Sigma v=\lambda v
\]

where

- \(v\) = Eigenvector (Principal Component)
- \(\lambda\) = Eigenvalue (Variance explained)

---

### 4. Projection

After selecting the top \(k\) eigenvectors,

\[
X_{new}=X_{centered}W
\]

where

- \(W\) = Projection Matrix
- \(X_{new}\) = Lower-dimensional representation

---

## Repository Structure

```
Principal_Component_Analysis/
│
├── pca.py
├── visualization.py
└── README.md
```

---

## Concepts Covered

- Mean Centering
- Variance
- Covariance
- Covariance Matrix
- Eigenvalues
- Eigenvectors
- Principal Components
- Orthogonal Projection
- Dimensionality Reduction

---

## Applications

- Feature Engineering
- Data Compression
- Noise Reduction
- Image Compression
- Data Visualization
- Preprocessing for Machine Learning

---

## Limitations

- PCA assumes directions with the highest variance contain the most useful information.
- Principal Components are combinations of original features, making them harder to interpret.
- Sensitive to feature scaling (features should usually be standardized before PCA).

---

## Future Improvements

- [ ] Explained Variance Ratio
- [ ] Scree Plot
- [ ] Inverse Transform
- [ ] Store Training Mean for Transforming New Data
- [ ] Whitening
- [ ] Animated Visualization
- [ ] Image Compression Example

---

## What I Learned

Implementing PCA from scratch helped connect several mathematical concepts that often seem unrelated:

- Variance from Statistics
- Covariance between features
- Matrix transformations from Linear Algebra
- Eigenvectors as directions of maximum variance
- Eigenvalues as the amount of variance explained
- Orthogonal projection into a lower-dimensional space

Rather than being a predictive algorithm, PCA is a feature transformation technique that finds a more informative coordinate system for representing data.