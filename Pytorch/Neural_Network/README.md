# PyTorch Neural Network

This project implements a **multi-layer neural network** using **PyTorch**.

The project builds on the neural-network implementations developed from scratch using NumPy and demonstrates how PyTorch automates many of the mathematical operations involved in training neural networks.

The objective was to understand how a neural network is structured in PyTorch, how multiple layers are constructed dynamically, and how **automatic differentiation** replaces the manual backpropagation calculations implemented previously.

---

# Objective

Given a dataset containing two input features and a continuous target:

| Feature 1 | Feature 2 | Target |
| --------: | --------: | -----: |
| 1 | 1 | 4 |
| 1 | 2 | 6 |
| 2 | 1 | 7 |
| 2 | 2 | 9 |
| 3 | 1 | 12 |
| 1 | 3 | 8 |
| 3 | 2 | 14 |
| 2 | 3 | 11 |

The neural network learns to predict the continuous target from the two input features.

This is therefore a **regression problem**.

---

# Mathematical Background

Each neuron performs a weighted sum of its inputs:

```text
z = Wx + b
```

The result is passed through an activation function.

For the hidden layers, the network uses the **ReLU** activation:

```text
ReLU(z) = max(0, z)
```

The final output layer uses a linear transformation because the network is performing regression:

```text
ŷ = Wx + b
```

---

# Network Architecture

The network uses multiple hidden layers containing five neurons each.

```text
Input
  |
Linear(2 → 5)
  |
ReLU
  |
Linear(5 → 5)
  |
ReLU
  |
...
  |
Linear(5 → 1)
  |
Prediction
```

The layers are constructed dynamically instead of manually creating `layer1`, `layer2`, `layer3`, and so on.

---

# Sequential

PyTorch's `nn.Sequential` is used to construct the sequence of layers.

```python
nn.Sequential(*layers)
```

The `*` performs Python argument unpacking.

`Sequential` then passes the output of each module into the next module.

Conceptually:

```text
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
Linear
```

This allows an arbitrary number of layers to be constructed without manually defining every layer.

---

# PyTorch Module

The neural network inherits from:

```python
nn.Module
```

The architecture is defined inside `__init__` and forward propagation is defined inside:

```python
forward()
```

Calling:

```python
n_network(x)
```

automatically invokes the model's `forward()` method.

---

# Loss Function

The model uses **Mean Squared Error**:

```python
nn.MSELoss()
```

Conceptually:

```text
MSE = mean((ŷ - y)²)
```

The loss measures how far the predictions are from the actual target values.

---

# Backpropagation

In the previous neural-network implementation, the gradients were calculated manually using the chain rule.

PyTorch provides **automatic differentiation** through its autograd system.

After calculating the loss:

```python
cost.backward()
```

PyTorch automatically applies the chain rule through the computation graph and calculates the gradients for the trainable parameters.

```text
Forward Propagation
        ↓
     Prediction
        ↓
       Loss
        ↓
   cost.backward()
        ↓
Calculate gradients
```

`backward()` calculates gradients but does not update the weights.

---

# Gradient Accumulation

PyTorch accumulates gradients by default.

Therefore, previous gradients must be cleared before calculating new ones:

```python
optimizer.zero_grad()
```

The training iteration therefore begins by resetting the gradients.

---

# Gradient Descent

The network uses **Stochastic Gradient Descent (SGD)**:

```python
torch.optim.SGD(
    n_network.parameters(),
    lr=0.001
)
```

The basic update is:

```text
parameter_new =
parameter_old - learning_rate × gradient
```

The gradients are calculated by:

```python
cost.backward()
```

The actual parameter update occurs with:

```python
optimizer.step()
```

---

# Training Loop

The complete training process follows:

```text
Clear previous gradients
        |
        v
Forward propagation
        |
        v
Calculate prediction
        |
        v
Calculate loss
        |
        v
Backpropagation
        |
        v
Calculate gradients
        |
        v
Update parameters
        |
        v
Repeat
```

The corresponding PyTorch operations are:

```python
optimizer.zero_grad()

prediction = n_network(x1)

cost = cost_function(prediction, y1)

cost.backward()

optimizer.step()
```

This process is repeated for multiple epochs.

---

# Epochs

The implementation trains the model for:

```python
700
```

epochs.

The loss can be monitored during training.

The objective is for the loss to decrease as the network learns the relationship between the inputs and targets.

---

# Prediction

After training, the network can predict the target for a new sample:

```python
x_new = torch.tensor([[1, 1]], dtype=torch.float32)

predict = n_network(x_new)
```

The network performs:

```text
Input
  ↓
Hidden Layers
  ↓
Output Layer
  ↓
Prediction
```

For:

```text
x = [1, 1]
```

the corresponding training target is:

```text
y = 4
```

---

# Deep Network Experiment

The project also experimented with increasing the number of hidden layers.

A very deep network was tested to understand how network depth affects training.

The experiment demonstrated that:

```text
More layers ≠ Automatically better model
```

For this small dataset, reducing the number of hidden layers resulted in a substantially lower cost.

Advanced initialization and optimization techniques were intentionally not introduced at this stage. These will be studied separately rather than being added without understanding them.

---

# Implementation

Main components include:

- Tensor creation
- Neural network definition
- Dynamic layer construction
- `nn.Linear`
- ReLU activation
- `nn.Sequential`
- Forward propagation
- MSE loss
- Automatic differentiation
- Gradient calculation
- Gradient clearing
- SGD optimization
- Parameter updates
- Epoch-based training
- Prediction

---

# Files

```text
neural_network/
│
├── neural_network.py
└── README.md
```

---

# Concepts Learned

- Neural Networks
- Artificial Neurons
- Weights
- Biases
- Linear Layers
- ReLU
- Multiple Hidden Layers
- `nn.Module`
- `nn.Linear`
- `nn.Sequential`
- Dynamic Layer Construction
- Forward Propagation
- Mean Squared Error
- Backpropagation
- Automatic Differentiation
- Autograd
- Gradient Accumulation
- `zero_grad()`
- `backward()`
- `optimizer.step()`
- Stochastic Gradient Descent
- Learning Rate
- Epochs
- Model Prediction

---

# Connection to the From-Scratch Implementation

The main purpose of this project is to connect the mathematical implementation to PyTorch.

Previously:

```text
Forward Propagation
        ↓
Calculate Loss
        ↓
Manually Apply Chain Rule
        ↓
Calculate Gradients
        ↓
Manually Update Parameters
```

With PyTorch:

```text
Forward Propagation
        ↓
Calculate Loss
        ↓
cost.backward()
        ↓
Autograd Calculates Gradients
        ↓
optimizer.step()
        ↓
Parameters Updated
```

The underlying mathematics has not disappeared. PyTorch is automating the calculations that were previously implemented manually.

---

# Future Improvements

- [ ] Explore different optimizers
- [ ] Study initialization methods
- [ ] Study optimization techniques
- [ ] Convolutional Neural Networks
- [ ] Attention mechanisms
- [ ] Transformers

---

This implementation is part of the **Machine Learning Foundations** repository, where algorithms are studied mathematically before being implemented and explored in Python.
