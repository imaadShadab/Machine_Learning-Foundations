# Softmax Regression from Scratch

This project implements **Softmax Regression** from scratch using **NumPy** and **Gradient Descent**.

Softmax Regression extends Logistic Regression from binary classification to **multi-class classification**. Instead of predicting the probability of a single class using the sigmoid function, Softmax Regression assigns a probability to every possible class using the **Softmax Function**.

The objective was to understand the mathematics behind multi-class classification before implementing the algorithm.

---

# Objective

Given a dataset containing two input features and three possible classes

| Feature 1 | Feature 2 | Class |
|---|---|---|
| 1 | 1 | 0 |
| 1 | 2 | 0 |
| 2 | 1 | 0 |
| 5 | 5 | 1 |
| 5 | 6 | 1 |
| 6 | 5 | 1 |
| 9 | 1 | 2 |
| 9 | 2 | 2 |
| 8 | 1 | 2 |

learn a model that predicts the probability that a new input belongs to each of the **three classes**.

For example, the classes could represent

```text
0 → Horse
1 → Cat
2 → Dog
```

while the two features could represent properties such as height and weight.

---

# Mathematical Background

Softmax Regression computes a separate linear score for every possible class.

For class `k`:

```text
zk = θkᵀx
```

where each class has its own parameter vector `θk`.

These scores are then converted into probabilities using the Softmax Function.

---

## Design Matrix

For two input features, the design matrix is represented as

```text
1  x₁₁  x₁₂
1  x₂₁  x₂₂
1  x₃₁  x₃₂
...
```

The first column contains ones for the intercept term.

If there are `m` training examples and `n` features, the design matrix has shape

```text
(m, n + 1)
```

---

## Parameter Matrix

Unlike binary Logistic Regression, which uses one parameter vector, Softmax Regression learns a parameter vector for every class.

The parameter vectors are stored as columns of a matrix.

```text
Θ =

[ θ₀₀  θ₀₁  θ₀₂ ]
[ θ₁₀  θ₁₁  θ₁₂ ]
[ θ₂₀  θ₂₁  θ₂₂ ]
```

Each column corresponds to one class.

For three design-matrix columns and three classes:

```text
Θ shape = (3, 3)
```

---

## Linear Scores

The scores for every training example and every class are computed simultaneously using

```text
Z = XΘ
```

where

- **X** is the design matrix
- **Θ** is the parameter matrix
- **Z** contains the class scores

For `m` examples and `K` classes:

```text
Z shape = (m, K)
```

Each row contains the scores for one training example across all classes.

---

## Softmax Function

The Softmax Function converts the class scores into probabilities.

For class `k`:

```text
P(y = k | x) = e^(θkᵀx) / Σⱼ e^(θjᵀx)
```

The exponential score for a class is divided by the sum of exponential scores across all classes.

Therefore, the probabilities for each training example sum to **1**.

Example:

```text
[0.10, 0.75, 0.15]
```

means the model assigns

```text
Class 0 → 10%
Class 1 → 75%
Class 2 → 15%
```

---

## One-Hot Encoding

The original class labels

```text
0
1
2
```

are converted into vectors.

```text
0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]
```

This produces an output matrix `Y` with the same shape as the probability matrix.

In NumPy:

```python
Y = np.eye(K_classes)[y]
```

---

## Cross-Entropy Loss

Softmax Regression uses multi-class cross-entropy loss.

```text
J(Θ) = -(1/m) Σᵢ Σₖ yᵢₖ log(pᵢₖ)
```

where

- **m** is the number of training examples
- **K** is the number of classes
- **yᵢₖ** indicates whether example `i` belongs to class `k`
- **pᵢₖ** is the predicted probability for class `k`

Because the true labels are one-hot encoded, only the probability assigned to the correct class contributes to the loss.

For example:

```text
Actual:
[0, 1, 0]

Prediction:
[0.10, 0.80, 0.10]
```

The loss for this example becomes

```text
-log(0.80)
```

Confident correct predictions produce a small loss, while assigning a low probability to the correct class produces a large loss.

---

## Residual Matrix

The prediction error is calculated using

```text
P - Y
```

where

- **P** is the matrix of predicted probabilities
- **Y** is the one-hot encoded output matrix

For example:

```text
Prediction:
[0.10, 0.70, 0.20]

Actual:
[0, 1, 0]

Residual:
[0.10, -0.30, 0.20]
```

The residual matrix contains the prediction errors for every example and every class.

---

## Gradient

The gradient of the cross-entropy loss simplifies to

```text
∇J(Θ) = (1/m) Xᵀ(P - Y)
```

where

- **X** is the design matrix
- **P** is the probability matrix
- **Y** is the one-hot encoded target matrix

The resulting gradient matrix has the same shape as the parameter matrix `Θ`.

---

## Gradient Descent

The parameter matrix is updated using

```text
Θ = Θ − α∇J(Θ)
```

where

- **α** is the learning rate

The process is repeated until the change in cost becomes sufficiently small or the maximum number of iterations is reached.

---

# Algorithm

1. Create the design matrix.
2. Determine the number of classes.
3. Initialize the parameter matrix to zero.
4. One-hot encode the output labels.
5. Compute the class scores using `XΘ`.
6. Apply the Softmax Function.
7. Compute the residual matrix `P - Y`.
8. Compute the gradient.
9. Update the parameter matrix.
10. Compute the cross-entropy loss.
11. Repeat until convergence.

---

# Implementation Notes

The implementation is fully vectorized using NumPy.

Main computations:

```python
scores = design_matrix @ theta_matrix

scores_exp = np.exp(scores)

probabilities = scores_exp / np.sum(
    scores_exp,
    axis=1,
    keepdims=True
)

residual_matrix = probabilities - Y_outputMatrix

gradient_matrix = (
    design_matrix.T @ residual_matrix
) / m

theta_matrix -= learning_rate * gradient_matrix
```

The cross-entropy loss is computed using

```python
cost = (-1 / m) * np.sum(
    Y_outputMatrix * np.log(probabilities)
)
```

Training stops when the change in loss becomes sufficiently small.

---

# Prediction

Once the model has learned the parameter matrix, predictions for new examples are made using the same forward computation.

```text
Xnew
 ↓
XnewΘ
 ↓
Softmax
 ↓
Class Probabilities
 ↓
argmax
 ↓
Predicted Class
```

The class with the highest probability is selected using

```python
predictions = np.argmax(probabilities, axis=1)
```

For example:

```text
[0.05, 0.15, 0.80]
```

produces

```text
Class 2
```

because class `2` has the highest predicted probability.

---

# Visualization

Since the dataset contains two features, the learned classifier can be visualized in two dimensions.

A grid of points is generated across the feature space using

```python
xx1, xx2 = np.meshgrid(...)
```

Every point in the grid is passed through the trained model.

The predicted class for each point is then displayed as a decision region.

The visualization includes

- Training samples
- Three predicted classes
- Learned decision regions
- Multi-class decision boundaries

This makes it possible to see how Softmax Regression divides the feature space between the different classes.

---

# Files

```text
softmax_regression/
│
├── softmaxRegression.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Multi-Class Classification
- Softmax Function
- Design Matrix
- Parameter Matrix
- Linear Class Scores
- One-Hot Encoding
- Multi-Class Cross-Entropy
- Residual Matrix
- Gradient Descent
- Decision Regions
- Argmax
- Matrix Multiplication
- NumPy Broadcasting
- Vectorization

---

# Future Improvements

- [ ] Numerical Stability for Softmax
- [ ] L2 Regularization
- [ ] Mini-batch Gradient Descent
- [ ] Stochastic Gradient Descent
- [ ] Feature Scaling
- [ ] Train/Test Split
- [ ] Accuracy Evaluation
- [ ] Confusion Matrix
- [ ] Support More Classes
- [ ] Support Higher-dimensional Datasets

---

# Connections

While implementing this algorithm, I found that Softmax Regression is a direct extension of Logistic Regression from binary classification to multi-class classification.

Instead of learning one parameter vector and producing one probability using the sigmoid function, Softmax Regression learns a parameter vector for every class and converts all class scores into a probability distribution using the Softmax Function.

The same mathematical ideas appear in several topics:

- Logistic Regression
- Multi-Class Classification
- Maximum Likelihood Estimation
- Cross-Entropy Loss
- Gradient Descent
- Generalized Linear Models
- Neural Networks
- Multi-Class Classification Layers

Softmax is especially important in neural networks, where it is commonly used to convert the output scores of a classification model into class probabilities.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.