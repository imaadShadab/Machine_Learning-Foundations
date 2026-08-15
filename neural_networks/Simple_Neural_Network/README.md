# Simple Neural Network — Linear Regression

A from-scratch implementation of a simple neural network for **linear regression**, built with NumPy.

## Overview

The network currently consists of a **single linear neuron**.

```text
Input Features
      |
      v
+-------------+
|   Neuron    |
|  z = XW + b |
+-------------+
      |
      v
 Prediction
      |
      v
    Loss
      |
      v
 Backpropagation
      |
      v
 Gradient Descent
      |
      +-------> Update W and b
```

Because the output neuron uses a linear activation,

`a = f(z) = z`

the network is mathematically equivalent to a linear regression model.

The purpose of implementing it as a neural network is to establish the foundations that will later be extended to multiple neurons and hidden layers.

## Dataset

The initial dataset follows:

`y = 2x + 1`

```python
x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [3, 5, 7, 9, 11, 13, 15, 17]
```

The network therefore needs to learn approximately:

`w = 2`

`b = 1`

## Model

The neuron calculates:

`z = XW + b`

Since this is a regression problem, the activation function is linear:

`y_hat = z`

Therefore:

`y_hat = XW + b`

## Loss Function

The model uses the squared-error loss:

`L = 1/2 (y_hat - y)^2`

For the full dataset, the implementation uses the mean of these losses.

The factor of `1/2` makes the derivative cleaner because the exponent's 2 cancels during differentiation.

## Backpropagation

Backpropagation determines how much each parameter contributed to the loss.

For the weight:

`dL/dw = (y_hat - y)x`

For the bias:

`dL/db = y_hat - y`

These gradients are calculated for each training example and then averaged before updating the parameters.

## Gradient Descent

The parameters are updated using:

`w_new = w_old - learning_rate * dL/dw`

`b_new = b_old - learning_rate * dL/db`

The learning rate controls the size of each update.

A learning rate that is too large can cause the loss to explode rather than converge.

## Training Loop

```text
1. Forward propagation
        |
2. Calculate prediction
        |
3. Calculate loss
        |
4. Backpropagation
        |
5. Calculate gradients
        |
6. Update weights and bias
        |
7. Repeat
```

The loss should decrease as the model learns the underlying relationship.

## Project Structure

```text
Simple_Neural_Network/
|
+-- math_functions.py
+-- regression_neural_network.py
+-- visualization.png
+-- README.md
```

### `regression_neural_network.py`

Contains the `RegressionNeuralNetwork` class and the main training workflow:

- Parameter initialization
- Forward propagation
- Loss calculation
- Backpropagation
- Gradient descent
- Training over multiple epochs

### `math_functions.py`

Contains the mathematical helper functions:

- Loss calculation
- Loss gradient
- Weight gradient
- Bias gradient

Keeping the mathematical operations separate makes the neural-network implementation easier to read.

### `visualization.png`

Visualization of the model and training concept.

## Technologies

- Python
- NumPy

No machine-learning framework is used. The model and its training procedure are implemented from scratch.

## What This Project Demonstrates

- Neurons
- Weights
- Bias
- Linear activation
- Forward propagation
- Loss functions
- Gradients
- Backpropagation
- Gradient descent
- Learning rate
- Epochs
- Parameter updates

## Next Step

The next version will move beyond a single linear neuron and introduce:

```text
Input Layer
     |
     v
Hidden Layer
     |
Multiple Neurons
     |
ReLU Activation
     |
     v
Output Neuron
     |
     v
Prediction
```

This will introduce weight matrices, multiple biases, hidden-layer activations, and backpropagation through multiple layers.
