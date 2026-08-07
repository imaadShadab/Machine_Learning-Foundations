# Machine Learning Foundations

> Implementing machine learning algorithms from first principles using mathematics, NumPy, and Python.

---

## About

This repository documents my journey of learning machine learning by studying the mathematics behind algorithms and then implementing them from scratch.

The goal is not simply to use machine learning libraries, but to understand what happens underneath them: how predictions are computed, how loss functions measure error, how gradients are formed, and how models learn from data.

Every implementation is built using **NumPy**, with **Matplotlib** used to visualize learned models, decision boundaries, and algorithm behavior whenever appropriate.

The mathematical foundations come primarily from:

- Linear Algebra
- Calculus
- Probability & Statistics
- Optimization

---

## Learning Philosophy

For each algorithm, I aim to understand:

- Mathematical intuition
- The role of each term in the model
- Cost / objective functions
- Optimization methods
- Matrix and vector formulations
- Recursive algorithms (where applicable)
- Implementation from scratch
- Visualization of model behavior
- Strengths and limitations

The emphasis is on understanding **why the algorithm works and how the mathematics translates into code**.

---

## Repository Roadmap

The repository is organized to build machine learning concepts progressively.

1. **Regression**
   - Linear Regression
   - Locally Weighted Regression
   - Logistic Regression
   - Softmax Regression

2. **Probabilistic Learning**
   - Bernoulli Naive Bayes

3. **Tree-Based Learning**
   - Decision Trees
   - Random Forests
   - Gradient Boosted Trees *(Coming Soon)*

4. **Unsupervised Learning**
   - K-Means Clustering
   - Principal Component Analysis (PCA)

5. **Neural Networks**
   - Perceptron
   - Feedforward Networks
   - Backpropagation

---

## Implemented Algorithms

| Algorithm                        | Status | Visualization |
| -------------------------------- | :----: | :-----------: |
| Linear Regression                |   ✅   |      ✅       |
| Locally Weighted Regression      |   ✅   |      ✅       |
| Logistic Regression              |   ✅   |      ✅       |
| Softmax Regression               |   ✅   |      ✅       |
| Bernoulli Naive Bayes            |   ✅   |      ✅       |
| Decision Tree                    |   ✅   |      ✅       |
| Random Forest                    |   ✅   |      ✅       |
| K-Means Clustering               |   ✅   |      ✅       |
| Principal Component Analysis (PCA) | ✅ |      ✅       |

---

## Current Progress

### Linear Regression
- [x] Design Matrix
- [x] Mean Squared Error
- [x] Batch Gradient Descent
- [x] Normal Equation
- [ ] Feature Scaling
- [ ] Polynomial Regression

### Locally Weighted Regression
- [x] Gaussian Weight Function
- [x] Weight Matrix
- [x] Weighted Least Squares
- [x] Normal Equation
- [x] Local Prediction
- [x] Visualization

### Logistic Regression
- [x] Binary Classification
- [x] Sigmoid Function
- [x] Cross-Entropy Loss
- [x] Gradient
- [x] Batch Gradient Descent
- [x] Decision Boundary
- [ ] Newton's Method

### Softmax Regression
- [x] Multi-Class Classification
- [x] Parameter Matrix
- [x] Softmax Function
- [x] One-Hot Encoding
- [x] Multi-Class Cross-Entropy
- [x] Gradient Descent
- [x] Multi-Class Prediction
- [x] Decision Region Visualization

### Probability & Statistical Foundations
- [x] Bernoulli Distribution
- [x] Gaussian Distribution
- [x] Expectation
- [x] Variance
- [x] Standard Deviation
- [x] Likelihood
- [x] Log-Likelihood
- [x] Maximum Likelihood Estimation
- [x] Bernoulli Naive Bayes
- [x] Exponential Family — Conceptual Understanding
- [x] Generalized Linear Models — Conceptual Understanding

### Tree
- [x] Decision Trees
- [x] Random Forests
- [ ] Gradient Boosted Trees

### Neural Networks
- [ ] Perceptron
- [ ] Forward Propagation
- [ ] Activation Functions
- [ ] Backpropagation
- [ ] Gradient Descent

### Unsupervised Learning
- [X] K-Means Clustering
- [X] Principal Component Analysis (PCA)

---

## Repository Structure

```text
Machine_Learning-Foundations/
│
├── supervised_learning/
│   ├── linear_regression/
│   ├── locally_weighted_regression/
│   ├── logistic_regression/
│   ├── softmax_regression/
│   ├── naive_bayes/
│   ├── decision_tree/
│   └── random_forest/
│
├── unsupervised_learning/
│   ├── K-means_Clustering/
│   └── PCA/
│
└── README.md
```

Each implemented algorithm contains its own README explaining the mathematical intuition, implementation details, and visualization.

---

## Technologies

- Python
- NumPy
- Matplotlib

Future additions may include:

- SciPy
- PyTorch
- Jupyter Notebooks

---

## Mathematical Foundations

The implementations in this repository build on concepts including:

### Linear Algebra

- Vectors and Matrices
- Matrix Multiplication
- Linear Systems
- Vector Spaces
- Orthogonality
- Vector Projection
- Covariance Matrices
- Eigenvalues
- Eigenvectors
- Principal Components
- Least Squares

### Calculus & Optimization

- Derivatives
- Partial Derivatives
- Gradients
- Gradient Descent
- Newton's Method

### Probability & Statistics

- Probability Distributions
- Expectation
- Variance
- Standard Deviation
- Covariance
- Likelihood
- Log-Likelihood
- Maximum Likelihood Estimation
- Exponential Family

### Computer Science Concepts

- Recursion
- Binary Trees
- Divide and Conquer
- Greedy Algorithms

---

## References

Resources used while building this repository include:

- Stanford CS229 – Machine Learning
- Gilbert Strang – *Introduction to Linear Algebra*
- MIT 18.06 – Linear Algebra
- 3Blue1Brown – *Essence of Linear Algebra*
- 3Blue1Brown – *Essence of Calculus*

---

## Goal

## Goal

The objective of this repository is to build machine learning algorithms from first principles while developing a deep mathematical understanding of how they work internally.

Rather than relying on high-level libraries, every implementation begins with the underlying mathematics before translating each equation into NumPy code.

By the end of the project, the repository will cover supervised learning, unsupervised learning, ensemble methods, and neural networks, forming a complete collection of foundational machine learning algorithms implemented from scratch.