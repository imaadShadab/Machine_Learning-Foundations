# Multiple Neurons Neural Network

A from-scratch implementation of a simple feedforward neural network for regression using Python and NumPy.

This project builds on the previous single-neuron regression model by introducing:

- Multiple input features
- Multiple neurons in a hidden layer
- ReLU activation
- A linear output neuron
- Backpropagation through the hidden layer
- Gradient descent for updating weights and biases

The goal of this project is to understand how a neural network works internally rather than relying on machine-learning libraries such as TensorFlow or PyTorch.

---

## Network Architecture

```text
2 Input Features
       ↓
3 Hidden Neurons
       ↓
ReLU Activation
       ↓
1 Output Neuron
       ↓
Linear Activation
       ↓
Prediction
```

The number of hidden neurons can be specified when creating the network:

```python
nn = NeuralNetwork(3)
```

---

## Dataset

The network is trained on a small regression dataset containing two input features:

```python
x = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [1, 3],
    [3, 2],
    [2, 3]
])
```

with corresponding target values:

```python
y = np.array([
    4,
    6,
    7,
    9,
    12,
    8,
    14,
    11
])
```

The dataset is intentionally small so that the calculations and behavior of the network can be easily inspected.

---

## Forward Propagation

### Hidden Layer

Each hidden neuron calculates:

```text
Z = XWᵀ + b
```

The result is passed through ReLU:

```text
A = ReLU(Z)
```

ReLU is:

```text
ReLU(x) = max(0, x)
```

### Output Layer

The hidden-layer activations are passed to the output neuron:

```text
ŷ = AW_outputᵀ + b_output
```

Since this is a regression problem, no nonlinear activation function is applied to the output.

---

## Cost Function

The network uses Mean Squared Error with a factor of `1/2`:

```text
C = 1/2 · mean((ŷ - y)²)
```

The cost measures how far the network's predictions are from the actual target values.

---

## Backpropagation

Backpropagation calculates how each parameter in the network affects the cost using the chain rule.

```text
Cost
 ↓
Output Layer
 ↓
Hidden Layer Activation
 ↓
ReLU
 ↓
Hidden Layer Weights & Biases
```

### Output Gradient

```text
∂C/∂ŷ = (ŷ - y) / n
```

where `n` is the number of training examples.

### Output Weight Gradient

```text
∂C/∂W_output = ∂C/∂ŷ · A
```

### Hidden Layer Activation Gradient

```text
∂C/∂A = ∂C/∂ŷ · W_output
```

### ReLU Gradient

```text
ReLU'(z) = 1    if z > 0
           0    if z ≤ 0
```

Therefore:

```text
∂C/∂Z = ∂C/∂A · ReLU'(Z)
```

### Hidden Layer Weights

For:

```text
Z = XW + b
```

we have:

```text
∂C/∂W = ∂C/∂Z · X
```

### Hidden Layer Biases

Since:

```text
∂Z/∂b = 1
```

we have:

```text
∂C/∂b = ∂C/∂Z
```

---

## Gradient Descent

Parameters are updated using:

```text
parameter = parameter - learning_rate × gradient
```

This is applied to:

- Hidden-layer weights
- Hidden-layer biases
- Output weights
- Output bias

---

## Training

The training loop repeatedly performs:

```text
Forward Propagation
        ↓
Calculate Cost
        ↓
Backpropagation
        ↓
Gradient Descent
        ↓
Repeat
```

Example:

```python
nn = NeuralNetwork(3)
nn.fit(x, y, 700)
```

The network randomly initializes its weights before training.

---

## Making Predictions

After training:

```python
x_new = np.array([
    [1, 1]
])

prediction = nn.predict(x_new)

print(prediction)
```

---

## Visualization

`visualization.py` uses Matplotlib to visualize:

1. The neural-network architecture.
2. Actual vs. predicted values.

These visualizations make it easier to understand the structure of the network and evaluate its predictions.

---

## Project Structure

```text
Multiple_Neuron_Neural_Network/
│
├── neural_network.py
├── visualization.py
└── README.md
```

---

## What I Learned

This implementation helped me understand:

- Matrix multiplication between layers
- How multiple neurons are represented using matrices
- Forward propagation
- Activation functions
- ReLU
- Cost calculation
- Backpropagation
- The chain rule
- Gradients
- Gradient descent
- How gradients flow backward through a network
- How matrix dimensions correspond to features and neurons

The main progression from the previous project was:

```text
Single Neuron
     ↓
Multiple Features
     ↓
Multiple Hidden Neurons
     ↓
Activation Function
     ↓
Backpropagation Through Hidden Layer
```

---

## Current Limitations

This implementation is intentionally simple and educational.

The network currently uses basic random weight initialization:

```python
np.random.randn(...) * 0.1
```

Because the hidden layer uses ReLU, different random initializations can sometimes result in different training outcomes.

More advanced initialization techniques can improve this behavior, but they are outside the scope of this implementation and will be explored separately when studying more advanced neural-network concepts.

---

## Technologies

- Python
- NumPy
- Matplotlib

No machine-learning frameworks are used. The forward propagation, cost calculation, backpropagation, and gradient descent are implemented manually.
