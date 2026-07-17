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
| 5 | 11|

learn the parameters of the model

\[
h_\theta(x)=\theta_0+\theta_1x
\]

using **Gradient Descent**.

---

# Mathematical Background

The model predicts values using

\[
\hat{y}=X\theta
\]

where

- **X** is the design matrix
- **θ** is the parameter vector
- **ŷ** is the prediction vector

---

## Design Matrix

Instead of writing

\[
\theta_0+\theta_1x
\]

for every sample, the dataset is represented as

```
1  x₁
1  x₂
1  x₃
...
```

which allows every prediction to be computed using one matrix multiplication.

---

## Residual Vector

The residual vector measures how far every prediction is from the actual value.

\[
e=X\theta-y
\]

If

```
prediction = 6
actual = 7
```

then

```
error = -1
```

The residual vector contains these errors for every training example.

---

## Cost Function

The algorithm minimizes the Sum of Squared Errors

\[
J(\theta)=\frac12\sum (X\theta-y)^2
\]

The cost tells us **how wrong** the current model is.

Lower cost means better parameters.

---

## Gradient

The gradient tells us how each parameter should change.

\[
\nabla J(\theta)=X^T(X\theta-y)
\]

Instead of guessing new parameters, the gradient points in the direction of steepest increase.

Moving in the opposite direction decreases the cost.

---

## Gradient Descent Update

Parameters are updated using

\[
\theta:=\theta-\alpha\nabla J(\theta)
\]

where

- α = learning rate

This process repeats until the cost stops changing significantly.

---

# Algorithm

1. Initialize θ with zeros
2. Compute predictions
3. Compute residuals
4. Compute cost
5. Compute gradient
6. Update θ
7. Repeat until convergence

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

```
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
- Vectorization
- Residual Vector
- Cost Function
- Gradient
- Gradient Descent
- Convergence
- Matrix Multiplication

---

# Future Improvements

- Multiple Linear Regression
- Feature Scaling
- Polynomial Regression
- Learning Curves
- L1/L2 Regularization
- Mini-batch Gradient Descent
- Stochastic Gradient Descent

---

This implementation is part of the **Machine Learning Foundations** repository, where every algorithm is derived from its mathematical foundations before being implemented in Python.