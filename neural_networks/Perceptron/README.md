# Perceptron from Scratch

This project implements a **Perceptron Classifier** from scratch using **NumPy**.

The Perceptron is one of the simplest neural learning algorithms and provides a useful bridge between classical machine learning algorithms and **neural networks**.

Unlike Linear and Logistic Regression, the Perceptron does not minimize a differentiable cost function using gradient descent. Instead, it updates its weights only when it makes an incorrect prediction.

The objective was to understand how a single artificial neuron learns a linear decision boundary before moving on to multi-layer neural networks.

---

# Objective

Given a dataset containing two features and two classes:

| Feature 1 | Feature 2 | Class |
| --------: | --------: | ----: |
| 1 | 1 | 0 |
| 2 | 1 | 0 |
| 1 | 2 | 0 |
| 2 | 2 | 0 |
| 3 | 2 | 0 |
| 6 | 5 | 1 |
| 7 | 6 | 1 |
| 6 | 7 | 1 |
| 8 | 7 | 1 |
| 9 | 8 | 1 |

The Perceptron learns weights and a bias that define a linear decision boundary separating the two classes.

---

# Mathematical Background

A Perceptron computes a weighted sum of its inputs:

```text
z = wᵀx + b
```

where:

- **x** is the input vector.
- **w** is the weight vector.
- **b** is the bias.
- **z** is the calculated score.

The score is then passed through a step function.

---

## Step Function

The Perceptron converts the score into a binary prediction:

```text
ŷ = 1    if z >= 0
ŷ = 0    if z < 0
```

The step function determines which side of the decision boundary a sample belongs to.

---

# Decision Boundary

The decision boundary occurs where the score is zero:

```text
wᵀx + b = 0
```

For two features:

```text
w₁x₁ + w₂x₂ + b = 0
```

Rearranging:

```text
x₂ = -(w₁x₁ + b) / w₂
```

This produces the linear boundary separating the two classes.

The learned weights and bias therefore directly determine the position and orientation of the decision boundary.

---

# Perceptron Learning Rule

The Perceptron does not use gradient descent on a loss function.

Instead, it follows a simple mistake-driven learning rule.

For every training example:

1. Calculate the score.
2. Make a prediction.
3. Check whether the prediction is correct.
4. If correct, leave the weights unchanged.
5. If incorrect, update the weights and bias.

The error is:

```text
error = y - ŷ
```

The weight update is:

```text
w ← w + η(error)x
```

The bias update is:

```text
b ← b + η(error)
```

where **η** is the learning rate.

---

## Why Does the Update Work?

The update directly moves the model in the direction needed to correct the mistake.

### Class 1 incorrectly predicted as 0

```text
y = 1
ŷ = 0

error = 1
```

Therefore:

```text
w ← w + ηx
b ← b + η
```

This increases the score of the misclassified point, making it more likely to be classified as class 1.

### Class 0 incorrectly predicted as 1

```text
y = 0
ŷ = 1

error = -1
```

Therefore:

```text
w ← w - ηx
b ← b - η
```

This decreases the score of the misclassified point, making it more likely to be classified as class 0.

---

# Why There Is No Gradient Descent

Linear Regression and Logistic Regression learn by minimizing a loss function.

The general process is:

```text
Calculate loss
      ↓
Calculate gradient
      ↓
Move parameters downhill
      ↓
Repeat
```

The Perceptron instead uses:

```text
Make prediction
      ↓
Was it correct?
      ↓
   No
   ↓
Calculate error
      ↓
Update weights
      ↓
Make next prediction
```

The Perceptron therefore uses a **mistake-driven update** rather than following the gradient of a smooth loss function.

---

# Learning Rate

The learning rate controls the size of each correction.

```text
w ← w + η(error)x
```

A larger learning rate produces larger changes to the decision boundary.

A smaller learning rate produces smaller changes.

For example:

```text
η = 1.0
```

produces a larger correction than:

```text
η = 0.1
```

---

# Epochs

One epoch represents one complete pass through the training dataset.

The algorithm repeatedly processes the dataset:

```text
Epoch 1
    ↓
Update weights for mistakes
    ↓
Epoch 2
    ↓
Update weights for mistakes
    ↓
Epoch 3
    ↓
...
```

This allows the Perceptron to continue correcting mistakes made during previous passes.

---

# Convergence

The Perceptron does not need a conventional cost function to determine whether it has converged.

Instead, the number of classification mistakes is counted during every epoch.

For example:

```text
Epoch 1 → 4 mistakes
Epoch 2 → 2 mistakes
Epoch 3 → 1 mistake
Epoch 4 → 0 mistakes
```

When an entire epoch produces:

```text
0 mistakes
```

the model has correctly classified every training sample.

For linearly separable data, the Perceptron Convergence Theorem guarantees that the algorithm will eventually find a separating boundary.

The Perceptron does not necessarily find the optimal boundary. It finds a boundary that separates the training classes.

---

# Prediction

For a new sample, the trained Perceptron calculates:

```text
z = wᵀx + b
```

and applies the step function:

```text
prediction = 1 if z >= 0
prediction = 0 otherwise
```

For example, if the learned parameters are:

```text
w = [0.5, 2.5]
b = -7.5
```

and the new sample is:

```text
x = [5, 1]
```

then:

```text
z = (5)(0.5) + (1)(2.5) - 7.5
  = -2.5
```

Therefore:

```text
prediction = 0
```

---

# Implementation

The implementation is built from scratch using NumPy.

Main components include:

- Weight initialization
- Bias initialization
- Weighted sum calculation
- Step activation function
- Mistake detection
- Perceptron weight update
- Bias update
- Epoch-based training
- Convergence detection
- Prediction on new samples

---

# Visualization

The repository includes a visualization that displays:

- Class 0 samples
- Class 1 samples
- The learned decision boundary

The decision boundary is calculated directly from the learned weights and bias:

```text
w₁x₁ + w₂x₂ + b = 0
```

This provides a visual representation of how the Perceptron modifies its parameters until it finds a boundary separating the classes.

---

# Files

```text
perceptron/
│
├── perceptron.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Perceptron
- Artificial Neuron
- Weighted Sum
- Bias
- Step Function
- Linear Decision Boundaries
- Mistake-Driven Learning
- Learning Rate
- Epochs
- Convergence
- Binary Classification
- Linear Separability
- Weight Updates

---

# Connection to Neural Networks

The Perceptron is the simplest form of an artificial neuron.

A single Perceptron performs:

```text
x
↓
wᵀx + b
↓
Activation
↓
ŷ
```

A neural network extends this idea by using multiple neurons and multiple layers.

A single neuron:

```text
z = wᵀx + b
```

becomes a layer of neurons:

```text
z = Wx + b
```

where **W** contains the weights for multiple neurons.

Multiple layers can then be composed:

```text
Input
  ↓
Wx + b
  ↓
Activation
  ↓
Wx + b
  ↓
Activation
  ↓
Output
```

Therefore, the Perceptron provides the conceptual foundation for understanding how a single neuron eventually becomes part of a multi-layer neural network.

---

# Perceptron vs Logistic Regression

| | Perceptron | Logistic Regression |
|---|---|---|
| Output | Binary class | Probability |
| Activation | Step function | Sigmoid |
| Loss required | No | Yes |
| Gradient descent | No | Yes |
| Learning mechanism | Mistake-driven update | Gradient-based update |
| Decision boundary | Linear | Linear |
| Can produce probabilities | No | Yes |

Both models learn a linear decision boundary, but they learn it using fundamentally different training procedures.

---

# Future Improvements

- [ ] Add support for arbitrary binary datasets
- [ ] Add accuracy calculation
- [ ] Add configurable convergence tolerance
- [ ] Add randomized sample ordering
- [ ] Multi-class Perceptron
- [ ] Neural Network from scratch
- [ ] Multiple neurons per layer
- [ ] Multiple hidden layers
- [ ] Backpropagation

---

# Connections

While implementing the Perceptron, several concepts from mathematics and computer science come together.

The same ideas form the foundation for neural networks:

- Vectors
- Dot Products
- Matrix Multiplication
- Linear Decision Boundaries
- Bias Terms
- Activation Functions
- Iterative Parameter Updates
- Binary Classification

The Perceptron is therefore a useful bridge between classical machine learning and neural networks.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.
