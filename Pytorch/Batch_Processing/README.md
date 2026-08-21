# PyTorch Neural Network — Mini-Batch Training

This project implements a configurable **multi-layer neural network** using **PyTorch**.

The purpose of this implementation is to move from manually implementing neural-network training with NumPy to understanding the tools PyTorch provides for building and training neural networks.

The project introduces PyTorch's `Dataset` and `DataLoader` abstractions, mini-batch training, automatic differentiation, optimizers, and the distinction between training and evaluation modes.

---

# Objective

Given a small regression dataset containing two input features, the neural network learns to predict a continuous output.

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

The model is trained using PyTorch with multiple hidden layers and ReLU activations.

---

# PyTorch Dataset

The training data is wrapped in a custom PyTorch `Dataset`.

A Dataset provides access to individual training examples by associating each input with its corresponding target.

```text
Dataset
   |
   +-- index 0 → (x₀, y₀)
   +-- index 1 → (x₁, y₁)
   +-- index 2 → (x₂, y₂)
   +-- ...
```

The implementation provides two required methods:

```python
__len__()
__getitem__()
```

`__len__()` returns the number of training examples.

`__getitem__(index)` returns the input and target corresponding to that index.

This is conceptually similar to pairing input and target arrays together, as was done when implementing algorithms with NumPy.

---

# DataLoader

The PyTorch `DataLoader` takes the Dataset and provides the data in batches.

For example:

```python
DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

With eight training examples and a batch size of two:

```text
Dataset
   |
   v
DataLoader
   |
   +-- Batch 1 → 2 samples
   +-- Batch 2 → 2 samples
   +-- Batch 3 → 2 samples
   +-- Batch 4 → 2 samples
```

The DataLoader also provides optional shuffling so that the training examples are presented in a different order between epochs.

Mini-batches allow large datasets to be processed in smaller groups rather than passing the entire dataset through the network at once.

---

# Model Architecture

The neural network is configurable through:

- Number of input features
- Number of hidden neurons
- Number of hidden layers
- Number of output features

The current model uses:

```text
Input Layer
    2 features
       |
       v
Hidden Layer
    5 neurons
       |
     ReLU
       |
       v
Hidden Layer
    5 neurons
       |
     ReLU
       |
       v
Hidden Layer
    5 neurons
       |
     ReLU
       |
       v
Hidden Layer
    5 neurons
       |
     ReLU
       |
       v
Output Layer
    1 neuron
       |
       v
Prediction
```

The architecture is constructed using PyTorch's `nn.Sequential`.

The number of hidden layers and neurons can be changed without manually defining every layer.

For example:

```python
model = create_model(
    input_features=2,
    hidden_neurons=5,
    hidden_layers=4,
    output_features=1
)
```

---

# Forward Propagation

Each `Linear` layer performs a matrix multiplication followed by a bias:

```text
z = XWᵀ + b
```

The hidden layers then apply the ReLU activation:

```text
a = ReLU(z)
```

The final output layer is linear because this is a regression problem.

Therefore:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
ReLU
  ↓
...
  ↓
Linear
  ↓
Prediction
```

---

# Loss Function

The model uses PyTorch's Mean Squared Error loss:

```python
nn.MSELoss()
```

MSE measures the difference between the model's predictions and the target values.

Conceptually:

```text
Prediction
     |
     v
Compare with target
     |
     v
Calculate error
     |
     v
Mean Squared Error
```

The loss provides the value that is differentiated during backpropagation.

---

# Backpropagation

Unlike the earlier neural-network implementation where the gradients were derived and calculated manually using NumPy, PyTorch automatically calculates the required gradients.

The training process calls:

```python
cost.backward()
```

PyTorch uses the computation graph created during the forward pass to calculate the gradients of the loss with respect to the model's parameters.

Conceptually:

```text
Forward Propagation
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
 Gradients for
 weights & biases
```

This is the same chain-rule-based backpropagation studied in the from-scratch neural-network implementation, but PyTorch performs the derivative calculations automatically.

---

# Optimizer

The model uses PyTorch's Stochastic Gradient Descent optimizer:

```python
torch.optim.SGD(
    model.parameters(),
    lr=0.001
)
```

The optimizer receives the model's parameters and uses their gradients to update them.

The update conceptually follows:

```text
parameter_new
=
parameter_old
-
learning_rate × gradient
```

The optimizer therefore performs the parameter-update portion of gradient descent after backpropagation has calculated the gradients.

---

# Mini-Batch Training

The training loop processes one batch at a time.

```text
Epoch
 |
 +-- Batch 1
 |     ↓
 |   Forward
 |     ↓
 |   Loss
 |     ↓
 |   Backpropagation
 |     ↓
 |   Parameter Update
 |
 +-- Batch 2
 |     ↓
 |   Forward
 |     ↓
 |   Loss
 |     ↓
 |   Backpropagation
 |     ↓
 |   Parameter Update
 |
 +-- ...
```

For the current dataset:

```text
8 samples
batch size = 2

1 epoch = 4 batches
```

Therefore, one epoch performs four optimizer updates.

This differs from the earlier full-batch implementation where the entire dataset was processed before a parameter update.

---

# Epoch Loss

Because the model now trains on multiple batches per epoch, the loss from the final batch does not represent the entire epoch.

The implementation therefore accumulates the loss from every batch:

```python
total_loss += cost.item()
```

and calculates the average:

```python
average_loss = total_loss / len(dataloader)
```

This provides an average training loss across all batches in the epoch.

---

# Training Mode

Before training, the model is switched to training mode:

```python
model.train()
```

This tells PyTorch that the model is being used for training.

Some PyTorch layers, such as Dropout and Batch Normalization, behave differently during training and evaluation.

The current network does not contain these layers, so there is no visible difference in its behavior, but using the correct mode establishes the standard PyTorch training workflow.

---

# Evaluation Mode

When evaluating the model, it is switched to evaluation mode:

```python
model.eval()
```

This tells layers that behave differently during training that the model is now being used for evaluation or inference.

Evaluation mode is separate from gradient tracking.

The implementation also uses:

```python
with torch.no_grad():
```

This disables gradient tracking because the model is not being trained during evaluation.

Therefore:

```text
model.eval()
    ↓
Switch model to evaluation mode

torch.no_grad()
    ↓
Disable gradient tracking
```

These perform two different jobs.

---

# Training vs Evaluation

The training workflow is:

```text
model.train()
      ↓
Forward propagation
      ↓
Calculate loss
      ↓
Backward propagation
      ↓
Update parameters
```

The evaluation workflow is:

```text
model.eval()
      ↓
Disable gradients
      ↓
Forward propagation
      ↓
Calculate loss
```

No backpropagation or optimizer update occurs during evaluation.

---

# Implementation

The implementation is organized into separate functions for the major parts of the workflow.

### `MyDataset`

Provides indexed access to the input and target pairs.

### `create_dataloader()`

Creates a PyTorch DataLoader with a configurable batch size and shuffle option.

### `create_model()`

Creates a configurable multi-layer neural network.

The architecture can be controlled through:

```text
input_features
hidden_neurons
hidden_layers
output_features
```

### `fit()`

Contains the training loop:

- Switch model to training mode
- Iterate through batches
- Clear gradients
- Perform forward propagation
- Calculate loss
- Perform backpropagation
- Update parameters
- Calculate average epoch loss

### `evaluate()`

Evaluates the model without changing its parameters:

- Switch model to evaluation mode
- Disable gradient tracking
- Generate predictions
- Calculate average loss

---

# Files

```text
Pytorch/
│
├── batch_processing.py
└── README.md
```

The Python file contains:

- Dataset implementation
- DataLoader creation
- Configurable neural-network architecture
- Training function
- Evaluation function
- Prediction workflow

---

# Concepts Learned

- PyTorch `Dataset`
- PyTorch `DataLoader`
- Mini-batch training
- Batch size
- Epochs
- `nn.Sequential`
- Configurable neural-network architectures
- `nn.Linear`
- ReLU activation
- Mean Squared Error
- Automatic differentiation
- `backward()`
- Optimizers
- SGD
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- Training loss
- Evaluation loss
- Modular training workflows

---

# From-Scratch Implementation vs PyTorch

The earlier neural-network implementation manually calculated the gradients using the chain rule.

The PyTorch implementation replaces those manual derivative calculations with automatic differentiation.

```text
From Scratch:

Forward
   ↓
Loss
   ↓
Manually derive gradients
   ↓
Calculate gradients
   ↓
Update parameters
```

With PyTorch:

```text
Forward
   ↓
Loss
   ↓
cost.backward()
   ↓
PyTorch calculates gradients
   ↓
optimizer.step()
   ↓
Update parameters
```

PyTorch automates the calculations that were previously implemented manually.

---

# Current Limitation

The current implementation uses the same dataset for both training and evaluation.

```text
Dataset
   |
   +------------------+
   |                  |
   v                  v
Training           Evaluation
```

This is acceptable for learning the mechanics of the PyTorch workflow, but it does not provide a meaningful measure of how well the model generalizes to unseen data.

A proper machine-learning workflow will separate the available data into different sets.

---

# Connection to the Practical ML Workflow

This project represents the transition from implementing individual algorithms from scratch to using a machine-learning framework.

The workflow is now:

```text
Dataset
   ↓
Dataset / DataLoader
   ↓
Model
   ↓
Training
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Evaluation
```

The next step is to introduce separate **training and validation data**, followed by a proper train/validation/test workflow.

This will establish the practical machine-learning workflow used when training real models.

---

# Next Step

The next stage will introduce:

```text
Complete Dataset
       |
       +----------------+
       |                |
       v                v
 Training Set      Validation Set
       |                |
       v                v
   Train Model       Evaluate
```

After that, a separate test set will be introduced for final evaluation on unseen data.

---

This implementation is part of the **Machine Learning Foundations** repository, where machine-learning concepts are studied mathematically before being implemented and then explored using modern frameworks such as PyTorch.