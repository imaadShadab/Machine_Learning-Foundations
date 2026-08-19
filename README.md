# Machine Learning Foundations

> Implementing machine learning algorithms from first principles using mathematics, NumPy, and Python.

---

## About

This repository documents my journey of learning machine learning by studying the mathematics behind algorithms and implementing them from scratch.

The goal is not simply to use machine learning libraries, but to understand what happens underneath them:

- How predictions are computed
- How objective and loss functions measure error
- How gradients are derived
- How optimization algorithms update parameters
- How mathematical formulations translate into vectorized NumPy code
- How different algorithms make different assumptions about data

Every implementation is built primarily using **NumPy**, with **Matplotlib** used to visualize learned models, decision boundaries, dimensionality reduction, and algorithm behavior.

The mathematical foundations behind the implementations include:

- Linear Algebra
- Calculus
- Probability & Statistics
- Optimization

---

## Learning Philosophy

For each algorithm, I aim to understand:

- Mathematical intuition
- Model formulation
- Role of each mathematical term
- Objective / loss function
- Optimization method
- Matrix and vector formulation
- Derivation of gradients where applicable
- Implementation from scratch
- Visualization of model behavior
- Strengths and limitations
- Computational considerations

The emphasis is on understanding **why an algorithm works**, not simply reproducing its implementation.

---

# Repository Roadmap

The repository progresses from fundamental statistical models toward neural networks.

### 1. Regression
- Linear Regression
- Locally Weighted Regression
- Logistic Regression
- Softmax Regression

### 2. Probabilistic Learning
- Bernoulli Naive Bayes

### 3. Tree-Based Learning
- Decision Trees
- Random Forests
- Gradient Boosted Trees

### 4. Unsupervised Learning
- K-Means Clustering
- Principal Component Analysis (PCA)

### 5. Neural Networks
- Perceptron
- Forward Propagation
- Activation Functions
- Loss Functions
- Backpropagation
- Gradient Descent
- Multilayer Neural Networks

### 6. Future Deep Learning
- Binary Classification Networks
- Multiclass Classification Networks
- Optimization Algorithms
- Convolutional Neural Networks
- Attention
- Transformers

---

# Implemented Algorithms

| Algorithm | Status | Visualization |
|---|:---:|:---:|
| Linear Regression | ✅ | ✅ |
| Locally Weighted Regression | ✅ | ✅ |
| Logistic Regression | ✅ | ✅ |
| Softmax Regression | ✅ | ✅ |
| Bernoulli Naive Bayes | ✅ | ✅ |
| Decision Tree | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| Gradient Boosting | ✅ | ✅ |
| K-Means Clustering | ✅ | ✅ |
| Principal Component Analysis (PCA) | ✅ | ✅ |
| Perceptron | ✅ | ✅ |
| Feedforward Neural Network | ✅ | ✅ |
| Backpropagation | ✅ | ✅ |
| Multilayer Neural Network | ✅ | ✅ |

---

# Current Progress

## Linear Regression

- [x] Design Matrix
- [x] Mean Squared Error
- [x] Batch Gradient Descent
- [x] Normal Equation
- [ ] Feature Scaling
- [ ] Polynomial Regression

---

## Locally Weighted Regression

- [x] Gaussian Weight Function
- [x] Weight Matrix
- [x] Weighted Least Squares
- [x] Normal Equation
- [x] Local Prediction
- [x] Visualization

---

## Logistic Regression

- [x] Binary Classification
- [x] Sigmoid Function
- [x] Cross-Entropy Loss
- [x] Gradient Derivation
- [x] Batch Gradient Descent
- [x] Decision Boundary
- [ ] Newton's Method

---

## Softmax Regression

- [x] Multi-Class Classification
- [x] Parameter Matrix
- [x] Softmax Function
- [x] One-Hot Encoding
- [x] Multi-Class Cross-Entropy
- [x] Gradient Descent
- [x] Multi-Class Prediction
- [x] Decision Region Visualization

---

## Probability & Statistical Foundations

- [x] Bernoulli Distribution
- [x] Gaussian Distribution
- [x] Expectation
- [x] Variance
- [x] Standard Deviation
- [x] Covariance
- [x] Likelihood
- [x] Log-Likelihood
- [x] Maximum Likelihood Estimation
- [x] Bernoulli Naive Bayes
- [x] Exponential Family — Conceptual Understanding
- [x] Generalized Linear Models — Conceptual Understanding

---

## Tree-Based Learning

### Decision Trees

- [x] Recursive Tree Construction
- [x] Feature / Threshold Selection
- [x] Impurity Measures
- [x] Recursive Splitting
- [x] Leaf Nodes
- [x] Prediction
- [x] Visualization

### Random Forests

- [x] Bootstrap Sampling
- [x] Random Feature Selection
- [x] Decision Tree Ensemble
- [x] Majority Voting
- [x] Prediction
- [x] Visualization

### Gradient Boosting

- [x] Sequential Tree Construction
- [x] Residual / Error Fitting
- [x] Learning Rate
- [x] Ensemble Prediction
- [x] Visualization

---

## Unsupervised Learning

### K-Means Clustering

- [x] Centroid Initialization
- [x] Distance Calculation
- [x] Cluster Assignment
- [x] Centroid Update
- [x] Iterative Optimization
- [x] Visualization

### Principal Component Analysis

- [x] Data Centering
- [x] Covariance Matrix
- [x] Eigenvalues
- [x] Eigenvectors
- [x] Principal Components
- [x] Dimensionality Reduction
- [x] Visualization

---

# Neural Networks

The neural-network section focuses on understanding how neural networks are constructed and trained mathematically before using frameworks such as PyTorch.

### Perceptron

- [x] Weighted Sum
- [x] Bias
- [x] Activation
- [x] Prediction
- [x] Weight Updates

### Feedforward Neural Network

- [x] Forward Propagation
- [x] Linear Layers
- [x] ReLU Activation
- [x] Output Layer
- [x] Mean Squared Error
- [x] Gradient Descent

### Backpropagation

- [x] Chain Rule
- [x] Output Gradients
- [x] Hidden-Layer Gradients
- [x] ReLU Derivative
- [x] Weight Gradients
- [x] Bias Gradients
- [x] Parameter Updates

### Multilayer Neural Network

- [x] Multiple Hidden Neurons
- [x] Matrix-Based Forward Propagation
- [x] Matrix-Based Backpropagation
- [x] ReLU Hidden Layers
- [x] Linear Output Layer
- [x] Gradient Descent
- [x] Prediction

---

# Mathematical Foundations

The implementations in this repository build on the following mathematical concepts.

## Linear Algebra

- Vectors
- Matrices
- Matrix Multiplication
- Linear Systems
- Vector Spaces
- Linear Independence
- Basis
- Dimension
- Orthogonality
- Vector Projection
- Least Squares
- Covariance Matrices
- Eigenvalues
- Eigenvectors
- Principal Components

### Future

- [ ] Singular Value Decomposition (SVD)
- [ ] Positive Definite Matrices
- [ ] Matrix Decompositions

---

## Calculus & Optimization

- Derivatives
- Partial Derivatives
- Chain Rule
- Gradients
- Gradient Descent
- Loss Functions
- Optimization
- Newton's Method
- Mini-Batch / Stochastic Gradient Descent
### Future

- [ ] Second Derivatives
- [ ] Hessian Matrix
- [ ] Newton's Method
- [ ] Jacobians
- [ ] Momentum
- [ ] Adam

---

## Probability & Statistics

- Probability Distributions
- Bernoulli Distribution
- Gaussian Distribution
- Expectation
- Variance
- Standard Deviation
- Covariance
- Conditional Probability
- Likelihood
- Log-Likelihood
- Maximum Likelihood Estimation
- Exponential Family
- Bayes' Theorem

### Future

- [ ] Bayes' Theorem
- [ ] Multivariate Gaussian
- [ ] MAP Estimation
- [ ] Law of Large Numbers
- [ ] Central Limit Theorem

---

## Computer Science Concepts

- Recursion
- Binary Trees
- Divide and Conquer
- Greedy Algorithms
- Object-Oriented Programming
- Vectorized Numerical Computation

---

# Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn

### Planned

- SciPy
- Jupyter
- PyTorch

---

# References

The main learning resources used throughout this repository include:

- Stanford CS229 — Machine Learning
- Gilbert Strang — Linear Algebra
- MIT 18.06 — Linear Algebra
- 3Blue1Brown — Essence of Linear Algebra
- 3Blue1Brown — Essence of Calculus
- 3Blue1Brown — Neural Networks
- StatQuest — Statistics, Probability, and Machine Learning

---

