# Train / Validation Split with PyTorch

This project extends the previous **PyTorch Neural Network — Mini-Batch Training** implementation by introducing a proper **training and validation workflow**.

The main objective is to understand how a dataset is divided into training and validation subsets, how validation is performed without updating the model, and how training and validation losses can be tracked throughout training.

---

# Objective

The previous implementation trained and evaluated the model using the same dataset.

This project changes the workflow to:

```text
Complete Dataset
       |
       ↓
Train / Validation Split
       |
   +---+---+
   |       |
   ↓       ↓
Train   Validation
   |       |
   ↓       ↓
Learn   Evaluate
```

The model is trained only on the training set.

The validation set is kept separate and is used to evaluate how well the trained model performs on data that was not used to update its parameters.

---

# Train / Validation Split

PyTorch's `random_split()` is used to divide the dataset into two subsets.

The dataset contains:

```text
24 samples
```

The split used in this project is:

```text
80% → Training
20% → Validation
```

which results in:

```text
Training     → 19 samples
Validation   → 5 samples
```

The split is performed directly on the Dataset.

This means the relationship between each input and its corresponding target remains intact:

```text
x[i] ↔ y[i]
```

---

# Why Validation Data?

Training loss tells us how well the model performs on the data it is actively learning from.

However, a low training loss does not necessarily mean that the model will perform well on unseen data.

Validation data provides a separate set of examples that are not used to update the model's parameters.

Therefore:

```text
Training Loss
     ↓
How well is the model fitting the training data?

Validation Loss
     ↓
How well is the model performing on unseen examples?
```

Comparing the two gives us a better indication of the model's ability to generalize.

---

# Training and Validation DataLoaders

Separate DataLoaders are created for the two subsets.

```text
Training Dataset
       ↓
Training DataLoader

Validation Dataset
       ↓
Validation DataLoader
```

The training DataLoader uses:

```text
batch_size = 4
shuffle = True
```

The validation DataLoader uses:

```text
batch_size = 2
shuffle = False
```

Shuffling is useful during training because it changes the order in which the training examples are presented to the model.

It is not necessary during validation because the model is not learning from the validation data.

---

# Training Loop

The training process now takes place once per epoch across all training batches.

```text
Epoch
  |
  +-- Training Batch 1
  |      ↓
  |    Forward
  |      ↓
  |    Loss
  |      ↓
  |    Backward
  |      ↓
  |    Update
  |
  +-- Training Batch 2
  |
  +-- Training Batch 3
  |
  +-- ...
  |
  ↓
Calculate Average Training Loss
```

The loss from every training batch is accumulated.

The average training loss is then calculated:

```text
Average Training Loss
=
Sum of Batch Losses
--------------------
Number of Batches
```

This produces one training-loss value for every epoch.

---

# Validation Loop

After the model has finished training on all batches for an epoch, it is evaluated using the validation set.

The validation process is:

```text
Validation Batch
       ↓
Forward Propagation
       ↓
Calculate Loss
       ↓
Next Validation Batch
       ↓
...
       ↓
Average Validation Loss
```

Unlike training, validation does not modify the model.

There is no:

```python
cost.backward()
```

and no:

```python
optimizer.step()
```

during validation.

---

# Training Mode vs Evaluation Mode

The model is explicitly switched between training and evaluation modes.

During training:

```python
model.train()
```

During validation:

```python
model.eval()
```

These modes are important because certain PyTorch layers behave differently during training and evaluation.

For example, layers such as Dropout and Batch Normalization have different behavior depending on the mode.

Although the current network does not use these layers, using `train()` and `eval()` establishes the correct PyTorch workflow.

---

# Inference Mode

During validation, gradients are not required because the model's parameters are not being updated.

The validation loop therefore uses:

```python
with torch.inference_mode():
```

This disables gradient tracking during the evaluation process.

The distinction is:

```text
model.eval()
      ↓
Switch model to evaluation behavior

torch.inference_mode()
      ↓
Disable gradient tracking
```

These perform different jobs and are both useful during inference and evaluation.

---

# Training vs Validation Workflow

The complete workflow for each epoch is:

```text
                    Epoch
                      |
          +-----------+-----------+
          |                       |
          ↓                       ↓
      TRAINING                VALIDATION
          |                       |
     model.train()            model.eval()
          |                       |
     Training batches        Validation batches
          |                       |
       Forward                  Forward
          ↓                       ↓
        Loss                     Loss
          ↓
     backward()
          ↓
  optimizer.step()
          |
          +-----------+
                      |
                      ↓
               Next Epoch
```

The important distinction is:

```text
Training
→ Changes model parameters

Validation
→ Only measures performance
```

---

# Tracking Loss

The project stores the average loss from every epoch.

Two lists are maintained:

```python
train_losses
val_losses
```

Conceptually:

```text
Epoch 0
   ↓
Training Loss ─────→ train_losses
Validation Loss ───→ val_losses

Epoch 1
   ↓
Training Loss ─────→ train_losses
Validation Loss ───→ val_losses

...
```

This gives us a complete history of how both losses changed during training.

---

# Training vs Validation Loss

The recorded losses can be compared to understand the model's behavior.

For example:

```text
Training Loss
      ↓
      ↓
      ↓
      ↓
      ↓

Validation Loss
      ↓
      ↓
      ↑
      ↑
      ↑
```

If training loss continues decreasing while validation loss begins increasing, the model may be **overfitting**.

The model is improving on the training data while becoming worse at generalizing to unseen data.

---

# Overfitting

Overfitting occurs when a model fits the training data too closely and does not generalize well to new examples.

A typical pattern is:

```text
Training Loss
      ↓
      ↓
      ↓
      ↓
      ↓

Validation Loss
      ↓
      ↓
      ↑
      ↑
      ↑
```

The training loss decreases because the model continues improving on the examples it sees during training.

The validation loss eventually increases because the model's learned parameters are becoming less useful for unseen examples.

Validation data therefore provides an important signal that cannot be obtained by looking at training loss alone.

---

# Loss Visualization

The recorded training and validation losses are visualized using Matplotlib.

The visualization plots:

```text
Training Loss
       vs
Validation Loss
```

against the number of epochs.

Conceptually:

```text
Loss
 ↑
 |
 |\
 | \
 |  \       Training Loss
 |   \________________
 |
 |    \____
 |         \____
 |              \___ Validation Loss
 |
 +------------------------→ Epoch
```

This makes it possible to visually inspect how the model learns and whether the validation performance begins to diverge from the training performance.

---

# Small Dataset Limitation

The dataset used in this project contains only:

```text
24 samples
```

with only:

```text
5 validation samples
```

This is sufficient for learning the mechanics of train/validation splitting, but it is too small to provide a statistically reliable measurement of generalization.

Because the validation set contains so few examples, the validation loss can fluctuate significantly between epochs.

For example, a small change in the prediction of only one validation sample can have a noticeable effect on the average validation loss.

The dataset is therefore being used to demonstrate the workflow rather than to build a meaningful predictive model.

---

# Implementation

The implementation adds the following components to the previous PyTorch workflow:

- Dataset splitting using `random_split()`
- Separate training and validation subsets
- Separate DataLoaders
- Training-loss aggregation
- Validation-loss aggregation
- Loss history
- Evaluation during every epoch
- `model.train()`
- `model.eval()`
- `torch.inference_mode()`
- Training vs validation loss visualization

The neural-network architecture and basic PyTorch training concepts were introduced in the previous project and are reused here.

---

# Files

```text
Pytorch/

│
├── batch_processing.py
├── visualization.py
└── README.md
```

### `batch_processing.py`

Contains:

- Dataset
- Train/validation split
- DataLoaders
- Neural-network architecture
- Training loop
- Validation loop
- Loss aggregation
- Prediction

### `visualization.py`

Imports the recorded training and validation losses and plots them against the training epochs.

### `README.md`

Documents the train/validation workflow and the concepts introduced in this project.

---

# Concepts Learned

- Training Set
- Validation Set
- `random_split()`
- Train/Validation Splitting
- Separate DataLoaders
- Training Loss
- Validation Loss
- Loss Aggregation
- Loss History
- `model.train()`
- `model.eval()`
- `torch.inference_mode()`
- Generalization
- Overfitting
- Training vs Validation Curves

---

# From Training to Model Evaluation

The previous project established how to train a neural network using PyTorch.

This project adds the first important separation between **learning** and **evaluation**:

```text
Previous:

Dataset
   ↓
Model
   ↓
Training
   ↓
Evaluation
```

Now:

```text
Dataset
   ↓
Train / Validation Split
   ↓
   +-------------------+
   |                   |
   ↓                   ↓
Training            Validation
   ↓                   ↓
Model Updates       Evaluation
   |                   |
   +---------+---------+
             |
             ↓
       Compare Losses
```

This is an important step toward the practical machine-learning workflow.

---

# Connection to the Practical ML Workflow

The workflow developed so far is becoming:

```text
Dataset
   ↓
Train / Validation Split
   ↓
Dataset / DataLoader
   ↓
Model
   ↓
Training
   ↓
Validation
   ↓
Compare Performance
```

The next major addition is a separate **test set**.

The complete workflow will eventually become:

```text
                    Dataset
                       |
          +------------+------------+
          |            |            |
          ↓            ↓            ↓
       Training    Validation      Test
          |            |            |
          ↓            ↓            ↓
       Learn        Evaluate       Final
     Parameters      / Tune       Evaluation
```

The test set will remain completely separate from training and model development and will be used only for the final evaluation of the chosen model.

---

This implementation is part of the **Machine Learning Foundations** repository, where machine-learning concepts are studied mathematically before being implemented and then explored using modern frameworks such as PyTorch.