# Locally Weighted Linear Regression (LWR) from Scratch

This project implements **Locally Weighted Linear Regression (LWR)** from scratch using **NumPy** and the **Normal Equation**.

Unlike ordinary Linear Regression, which learns one global model for the entire dataset, LWR fits a new linear model every time a prediction is requested. Nearby training examples receive higher importance, while distant examples contribute less.

The objective was to understand the mathematics behind weighted least squares before implementing the algorithm.

---

# Objective

Given a dataset

| x | y |
|---|---|
| 1 | 1.2 |
| 2 | 1.8 |
| 3 | 3.0 |
| 4 | 4.5 |
| 5 | 6.8 |
| 6 | 6.5 |
| 7 | 5.0 |
| 8 | 4.0 |
| 9 | 3.8 |
|10 | 4.2 |

predict the output for a query point **x** by fitting a local linear model centered around that query.

---

# Mathematical Background

Unlike ordinary Linear Regression, the parameters are **not learned once**.

Instead, for every query point, a new parameter vector is computed.

```text
hθ(x) = θ₀ + θ₁x
```

where θ depends on the location of the query point.

---

## Design Matrix

The input data is represented as

```text
1  x₁
1  x₂
1  x₃
...
```

allowing predictions to be computed using matrix multiplication.

---

## Gaussian Weight Function

Each training example receives a weight based on its distance from the query point.

```text
w(i) = exp(-(x(i) - xquery)² / (2τ²))
```

where

- **τ** is the bandwidth parameter.
- Smaller τ focuses on nearby samples.
- Larger τ behaves more like ordinary Linear Regression.

Examples close to the query point receive weights near **1**, while distant points receive weights close to **0**.

---

## Weight Matrix

The individual weights are placed on the diagonal of a matrix.

```text
W =
[w₁ 0  0  ...]
[0  w₂ 0  ...]
[0  0  w₃ ...]
...
```

Since every training example is weighted independently, only the diagonal contains non-zero values.

---

## Weighted Least Squares

Instead of minimizing the ordinary squared error,

```text
Σ(Xθ − y)²
```

LWR minimizes the weighted objective

```text
Σ w(i)(Xθ − y)²
```

Errors made on nearby samples contribute much more to the optimization than errors made on distant samples.

---

## Normal Equation

The optimal parameters for the current query point are computed using

```text
θ = (XᵀWX)⁻¹XᵀWy
```

where

- **X** is the design matrix
- **W** is the diagonal weight matrix
- **y** is the target vector

A new θ is computed for every prediction request.

---

# Algorithm

For each query point:

1. Compute the distance to every training example.
2. Compute Gaussian weights.
3. Construct the diagonal weight matrix.
4. Solve the weighted normal equation.
5. Predict the output for the query point.

Repeat this process for every new query.

---

# Implementation Notes

The implementation is fully vectorized using NumPy.

Main computations:

```python
weights = np.exp(-((x - query) ** 2) / (2 * bandwidth ** 2))

W = np.diag(weights)

theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)

prediction = np.array([1, query]) @ theta
```

No optimization is performed beforehand.

The model is fit only when a prediction is requested.

---

# Visualization

To visualize the regression curve, predictions are computed over many evenly spaced query points.

```python
query_points = np.linspace(min(x), max(x), 200)
```

For each query point:

1. Compute a new local model.
2. Predict the corresponding output.
3. Store the prediction.

The resulting predictions are connected to form the locally weighted regression curve.

---

# Files

```text
local_weighted_regression/
│
├── localWeightRegression.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Design Matrix
- Gaussian Kernel
- Bandwidth (τ)
- Weight Matrix
- Weighted Least Squares
- Normal Equation
- Local Models
- Lazy Learning
- Matrix Multiplication
- Vectorization

---

# Future Improvements

- [ ] Higher-dimensional LWR
- [ ] Different Kernel Functions
- [ ] Automatic Bandwidth Selection
- [ ] KD-Tree / Ball Tree Optimization
- [ ] Comparison with Polynomial Regression
- [ ] Compare Different Bandwidth Values
- [ ] Performance Analysis on Large Datasets

---

# Connections

While implementing this algorithm, I found that Locally Weighted Regression extends Ordinary Least Squares by assigning different importance to each training example.

The same mathematical ideas appear in several topics:

- Linear Regression
- Weighted Least Squares
- Gaussian Kernels
- Kernel Methods
- Distance-based Learning
- Non-parametric Regression

These are different perspectives on fitting models that emphasize local structure in the data.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.