# Logistic Regression from Scratch

This project implements **Logistic Regression** from scratch using **NumPy** and **Gradient Descent**.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the probability that an example belongs to a particular class. The model learns a linear decision boundary and maps its output through the **Sigmoid Function**, producing probabilities between **0** and **1**.

The objective was to understand the mathematics behind binary classification before implementing the algorithm.

---

# Objective

Given a dataset

| x | y |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 1 |
| 6 | 1 |
| 7 | 1 |
| 8 | 1 |

learn a model that predicts the probability that a new input belongs to class **1**.

---

# Mathematical Background

Logistic Regression models a linear combination of the input features.

```text
z = θ₀ + θ₁x
```

Unlike Linear Regression, this value is not used directly as the prediction.

Instead, it is passed through the **Sigmoid Function** to obtain a probability.

---

## Design Matrix

The input data is represented as

```text
1  x₁
1  x₂
1  x₃
...
```

allowing predictions to be computed efficiently using matrix multiplication.

---

## Linear Model

The linear score is computed as

```text
z = Xθ
```

where

- **X** is the design matrix
- **θ** contains the model parameters

The score can take any real value.

---

## Sigmoid Function

The sigmoid function transforms the linear score into a probability.

```text
hθ(x) = 1 / (1 + e⁻ᶻ)
```

The output is always between **0** and **1**.

- Values near **0** indicate high confidence in class **0**.
- Values near **1** indicate high confidence in class **1**.
- Values near **0.5** indicate uncertainty.

---

## Cross-Entropy Loss

Instead of minimizing squared error, Logistic Regression minimizes the binary cross-entropy loss.

```text
J(θ) = -(1/m) Σ [ y log(hθ(x)) + (1-y) log(1-hθ(x)) ]
```

where

- **m** is the number of training examples
- **y** is the true class label
- **hθ(x)** is the predicted probability

This loss heavily penalizes confident incorrect predictions.

---

## Gradient

The gradient of the loss function simplifies to

```text
∇J(θ) = (1/m) Xᵀ(h - y)
```

where

- **h** is the vector of predicted probabilities
- **y** is the vector of true labels

The gradient indicates how each parameter should change to reduce the loss.

---

## Gradient Descent

The parameters are updated iteratively using

```text
θ = θ − α∇J(θ)
```

where

- **α** is the learning rate

This process is repeated until the model converges.

---

# Algorithm

1. Initialize the parameters to zero.
2. Compute the linear score.
3. Apply the sigmoid function.
4. Compute the prediction error.
5. Compute the gradient.
6. Update the parameters.
7. Compute the loss.
8. Repeat until convergence.

---

# Implementation Notes

The implementation is fully vectorized using NumPy.

Main computations:

```python
z = X @ theta

probabilities = sigmoid(z)

residual_vector = probabilities - y

gradient = (X.T @ residual_vector) / len(y)

theta -= learning_rate * gradient
```

The loss is computed using binary cross-entropy after each iteration.

Training stops when the change in loss becomes sufficiently small.

---

# Visualization

To visualize the learned classifier, the sigmoid curve is evaluated over many evenly spaced input values.

```python
x_plot = np.linspace(min(x), max(x), 300)
```

For each point:

1. Compute the linear score.
2. Apply the sigmoid function.
3. Plot the predicted probability.

The visualization also includes

- Training samples
- Learned sigmoid curve
- Decision boundary
- Probability threshold

---

# Files

```text
logistic_regression/
│
├── logistic_regression.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Binary Classification
- Design Matrix
- Linear Model
- Sigmoid Function
- Probability Estimation
- Cross-Entropy Loss
- Gradient Descent
- Decision Boundary
- Matrix Multiplication
- Vectorization

---

# Future Improvements

- [ ] Multi-class Logistic Regression (Softmax Regression)
- [ ] L2 Regularization (Ridge)
- [ ] L1 Regularization (Lasso)
- [ ] Mini-batch Gradient Descent
- [ ] Stochastic Gradient Descent
- [ ] Feature Scaling
- [ ] Performance Metrics (Accuracy, Precision, Recall, F1)
- [ ] ROC Curve and AUC
- [ ] Support Multi-dimensional Features

---

# Connections

While implementing this algorithm, I found that Logistic Regression builds directly upon Linear Regression by using a linear model to produce a score, then transforming that score into a probability using the sigmoid function.

The same mathematical ideas appear in several topics:

- Linear Regression
- Binary Classification
- Maximum Likelihood Estimation
- Gradient Descent
- Cross-Entropy Loss
- Neural Networks (Single Neuron)
- Generalized Linear Models

These concepts form the mathematical foundation for many modern machine learning algorithms.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.