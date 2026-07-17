# Linear Regression from Scratch

This project implements **Linear Regression** from scratch using **Batch Gradient Descent** and **NumPy**.

The objective was not simply to fit a line through data, but to understand the mathematics behind every step of the algorithm before writing the code.

---

# Objective

Given a dataset

| x | y |
|---|---|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

learn the parameters of the model

```text
hθ(x) = θ₀ + θ₁x
```

using **Gradient Descent**.

---

# Mathematical Background

The model predicts values using

```text
ŷ = Xθ
```

where

- **X** is the design matrix
- **θ** is the parameter vector
- **ŷ** is the prediction vector

---

## Design Matrix

Instead of writing

```text
θ₀ + θ₁x
```

for every sample, the dataset is represented as

```text
1  x₁
1  x₂
1  x₃
...
```

which allows every prediction to be computed using one matrix multiplication.

---

## Residual Vector

The residual vector measures how far every prediction is from the actual value.

```text
e = Xθ − y
```

If

```text
prediction = 6
actual = 7
error = -1
```

then the residual vector contains these errors for every training example.

---

## Cost Function

The algorithm minimizes the **Sum of Squared Errors**.

```text
J(θ) = ½ Σ(Xθ − y)²
```

The cost tells us **how wrong** the current model is.

Lower cost means better parameters.

---

## Gradient

The gradient tells us how each parameter should change.

```text
∇J(θ) = Xᵀ(Xθ − y)
```

Instead of guessing new parameters, the gradient points in the direction of steepest increase.

Moving in the opposite direction decreases the cost.

---

## Gradient Descent Update

Parameters are updated using

```text
θ ← θ − α∇J(θ)
```

where

- **α** = learning rate

This process repeats until the cost stops changing significantly.

---

# Algorithm

1. Initialize **θ** with zeros.
2. Compute predictions.
3. Compute the residual vector.
4. Compute the cost.
5. Compute the gradient.
6. Update the parameters.
7. Repeat until convergence.

---

# Implementation Notes

The implementation is fully vectorized using NumPy.

No loops are used over individual training examples.

Main computations:

```python
residual = X @ theta - y

gradient = X.T @ residual

theta = theta - learning_rate * gradient
```

---

# Files

```text
linear_regression/
│
├── linear_regression.py
└── README.md
```

---

# Concepts Learned

- Design Matrix
- Bias Term
- Linear Model
- Matrix Multiplication
- Vectorization
- Residual Vector
- Cost Function
- Gradient
- Gradient Descent
- Convergence

---

# Future Improvements

- [ ] Multiple Linear Regression
- [ ] Feature Scaling
- [ ] Polynomial Regression
- [ ] Learning Curves
- [ ] L1/L2 Regularization
- [ ] Mini-batch Gradient Descent
- [ ] Stochastic Gradient Descent

---

# Connections

While implementing this algorithm, I found that several concepts from different courses describe the same underlying mathematics:

- Least Squares
- Orthogonal Projection
- Normal Equation
- Linear Regression
- Minimizing Squared Error

These are different perspectives on the same optimization problem.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.