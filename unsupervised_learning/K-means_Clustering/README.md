# K-Means Clustering from Scratch

This project implements **K-Means Clustering** from scratch using **NumPy**.

Unlike supervised learning algorithms, K-Means learns patterns from **unlabeled data** by grouping similar observations into clusters. Instead of predicting predefined classes, it discovers structure within the dataset based on the distance between data points.

The objective was to understand the mathematics behind clustering, centroid optimization, Euclidean distance, and iterative optimization before implementing the algorithm.

---

# Objective

Given an unlabeled dataset

| x | y |
|---|---|
| 1 | 2 |
| 2 | 1 |
| 2 | 3 |
| 3 | 2 |
| 8 | 8 |
| 9 | 7 |
| 8 | 9 |
| 9 | 9 |
| 2 | 9 |
| 3 | 8 |
| 1 | 8 |
| 2 | 7 |

partition the data into **K clusters**, where each data point belongs to the cluster with the nearest centroid.

---

# Mathematical Background

K-Means aims to partition data into **K disjoint clusters** by minimizing the distance between each data point and the centroid of its assigned cluster.

---

## Cluster Centroid

Each cluster is represented by its **centroid**, which is simply the arithmetic mean of all points assigned to that cluster.

For a cluster containing

```text
x₁, x₂, ..., xₙ
```

its centroid is

```text
          x₁ + x₂ + ... + xₙ
μ = -------------------------
                n
```

The centroid represents the "center" of the cluster.

---

## Euclidean Distance

To determine which cluster a point belongs to, the Euclidean distance between the point and every centroid is computed.

```text
d(x,c) = √Σ(xᵢ − cᵢ)²
```

The point is assigned to the centroid with the smallest distance.

---

## Cluster Assignment

For every observation,

1. Compute its distance to every centroid.
2. Find the closest centroid.
3. Assign the observation to that cluster.

This creates a partition of the dataset into K clusters.

---

## Updating Centroids

After assigning every point, each centroid is recomputed as the mean of the points currently assigned to its cluster.

This process moves the centroid toward the center of its cluster.

---

## Objective Function

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**.

```text
Σ ||x − μ||²
```

The objective is to minimize the total squared distance between every point and the centroid of its assigned cluster.

Lower WCSS corresponds to tighter, more compact clusters.

---

## Convergence

The algorithm alternates between

- assigning points to clusters
- updating centroids

until the centroids stop changing.

Once the centroids remain unchanged, further iterations produce identical assignments, and the algorithm has converged.

---

# Algorithm

1. Randomly initialize K centroids.
2. Compute the distance from every point to every centroid.
3. Assign each point to its nearest centroid.
4. Compute the new centroid of every cluster.
5. Repeat Steps 2–4 until the centroids no longer change.

---

# Implementation Notes

The implementation is written using an object-oriented design.

The model stores

- learned centroids
- cluster assignments
- training data

during optimization.

Distance computation is performed using

```python
distance = np.sqrt(np.sum((data_point - centre) ** 2))
```

Cluster assignment repeatedly selects the nearest centroid

```python
closest_cluster = np.argmin(distances)
```

Centroids are then updated by averaging every cluster

```python
centroid = np.mean(cluster, axis=0)
```

Training stops once consecutive centroid locations are identical or the maximum number of iterations is reached.

---

# Visualization

A Matplotlib visualization is included to display

- discovered clusters
- learned centroids

An animated visualization is also provided, showing the complete optimization process as centroids move after every iteration until convergence.

---

# Files

```text
k_means/
│
├── kMeans.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Unsupervised Learning
- Clustering
- Euclidean Distance
- Cluster Centroid
- Arithmetic Mean
- Iterative Optimization
- Within-Cluster Sum of Squares (WCSS)
- Convergence
- Object-Oriented Design
- Vector Geometry

---

# Future Improvements

- [ ] K-Means++ Initialization
- [ ] Elbow Method
- [ ] Silhouette Score
- [ ] Mini-Batch K-Means
- [ ] Support Higher-Dimensional Visualization
- [ ] Improved Empty Cluster Handling

---

# Connections

While implementing this algorithm, I found that K-Means differs fundamentally from supervised learning algorithms.

Instead of learning from labeled examples, it discovers structure by repeatedly assigning observations to the nearest centroid and updating the centroid locations.

Several mathematical concepts connect directly to the implementation:

- Euclidean Geometry
- Vector Distance
- Arithmetic Mean
- Optimization
- Iterative Algorithms
- Voronoi Partitions
- Unsupervised Learning

Although K-Means is often introduced as a simple clustering algorithm, it illustrates many core ideas that appear throughout machine learning, including optimization, objective functions, convergence, and iterative model refinement.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.