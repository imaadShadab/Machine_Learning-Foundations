# Gradient Boosting

This project implements **Gradient Boosting for Regression** from scratch using **NumPy**, **Python**, and **scikit-learn's `DecisionTreeRegressor`** as the weak learner.

Unlike models such as Linear Regression, Gradient Boosting does not learn one set of parameters in a single optimization process. Instead, it builds an ensemble of weak regression trees sequentially, where each new tree attempts to correct the errors made by the previous model.

The objective was to understand how Gradient Boosting progressively improves predictions by learning from residuals and combining multiple weak learners.

---

# Objective

Given a dataset

| X | y |
| -: | -: |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |
| 6 | 13 |
| 7 | 15 |
| 8 | 17 |

build a regression model that progressively improves its predictions by training each new tree on the residuals of the current model.

---

# Mathematical Background

Gradient Boosting builds an additive model by sequentially adding weak learners.

The model begins with an initial prediction and then repeatedly adds corrections produced by regression trees.

---

## Initial Prediction

For Mean Squared Error, the initial prediction is the mean of the target values.

$$
F_0(x)=\frac{1}{n}\sum_{i=1}^{n}y_i
$$

This gives every training example the same initial prediction.

---

## Residuals

After making predictions, the residual for each training example is calculated as

$$
r_i=y_i-F(x_i)
$$

The residual represents the error that remains after the current model's prediction.

For Mean Squared Error, the residual is also the negative gradient of the loss with respect to the prediction.

Therefore, the next tree is trained to approximate these residuals.

---

## Training a Weak Learner

A regression tree is trained using

$$
X\rightarrow\text{residuals}
$$

The tree therefore does not directly predict the original target `y`.

Instead, it learns how the current model should be corrected in different regions of the feature space.

The regression tree used in this implementation is provided by scikit-learn:

```python
DecisionTreeRegressor
```

This allows the implementation to focus on the Gradient Boosting algorithm itself rather than implementing another regression tree.

---

## Learning Rate

The prediction of the new tree is multiplied by a learning rate before being added to the current model.

$$
F_m(x)=F_{m-1}(x)+\eta f_m(x)
$$

where:

- **\(F_m(x)\)** is the updated model.
- **\(F_{m-1}(x)\)** is the previous model.
- **\(f_m(x)\)** is the prediction of the new regression tree.
- **\(\eta\)** is the learning rate.

A smaller learning rate makes each tree contribute a smaller correction.

---

## Sequential Learning

The important idea behind Gradient Boosting is that every tree learns from the current model's remaining error.

```text
Initial Prediction
        ↓
Calculate Residuals
        ↓
Train Regression Tree
        ↓
Predict Residuals
        ↓
Multiply by Learning Rate
        ↓
Update Predictions
        ↓
Calculate New Residuals
        ↓
Train Next Tree
        ↓
        ...
```

Therefore, the trees are not independent.

Each tree builds on the corrections made by the previous trees.

---

# Example

Suppose the initial model predicts

```text
[5, 5, 5, 5]
```

while the actual targets are

```text
[2, 4, 6, 8]
```

The residuals are

```text
[-3, -1, 1, 3]
```

A regression tree is trained to predict these residuals.

Suppose the tree predicts

```text
[-2, -2, 2, 2]
```

and the learning rate is

```text
η = 0.5
```

The updated predictions become

```text
[5, 5, 5, 5]

+

0.5 × [-2, -2, 2, 2]

=

[4, 4, 6, 6]
```

The new residuals are then calculated using the original targets:

```text
[2, 4, 6, 8] − [4, 4, 6, 6]

=

[-2, 0, 0, 2]
```

The next tree learns from these new residuals.

This process continues for each tree in the ensemble.

---

# Prediction

Once training is complete, predictions for new data are calculated by adding the contribution of every trained tree to the initial prediction.

$$
F(x)=F_0(x)+\eta f_1(x)+\eta f_2(x)+\cdots+\eta f_M(x)
$$

For a new sample, every tree provides a correction based on the region of the feature space that the sample falls into.

The final prediction is the sum of the initial prediction and all of these corrections.

---

# Algorithm

1. Calculate the mean of `y` as the initial prediction.
2. Calculate the residuals between the actual targets and current predictions.
3. Train a regression tree using the residuals as the target.
4. Predict the residuals using the newly trained tree.
5. Multiply the tree predictions by the learning rate.
6. Add the correction to the current predictions.
7. Store the trained tree.
8. Repeat for the specified number of trees.
9. For new data, sum the initial prediction and the contribution of every trained tree.

---

# Implementation Notes

The Gradient Boosting algorithm is implemented manually.

The implementation handles:

- Initial mean prediction
- Residual calculation
- Sequential tree training
- Learning rate updates
- Storing trained trees
- Predictions using the complete ensemble

`DecisionTreeRegressor` from scikit-learn is used as the regression weak learner.

This was done intentionally because the focus of this implementation is understanding **Gradient Boosting**, rather than reimplementing the regression tree that acts as its weak learner.

---

# Hyperparameters

The current implementation uses:

```text
Maximum Trees = 20
Learning Rate = 0.5
Maximum Tree Depth = 2
```

### Maximum Trees

Controls the number of weak learners added to the ensemble.

### Learning Rate

Controls the size of each correction made by a tree.

### Maximum Tree Depth

Controls the complexity of each individual regression tree.

A shallow tree is used so that individual trees act as weak learners.

---

# Visualization

The repository includes a separate visualization using a nonlinear regression dataset.

The visualization displays:

- Training data
- Final Gradient Boosting predictions

The visualization demonstrates how multiple shallow regression trees can be combined to approximate a more complex relationship.

---

# Files

```text
gradient_boosting/
│
├── gradient_boosting.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Gradient Boosting
- Ensemble Learning
- Weak Learners
- Residuals
- Negative Gradients
- Mean Squared Error
- Regression Trees
- Learning Rate
- Additive Models
- Sequential Learning
- Function Approximation

---

# Future Improvements

- [ ] Implement early stopping
- [ ] Experiment with different learning rates
- [ ] Experiment with different tree depths
- [ ] Implement Gradient Boosting for classification
- [ ] Implement regression trees from scratch
- [ ] Compare against scikit-learn's Gradient Boosting implementation

---

# Connections

While implementing this algorithm, I found that Gradient Boosting connects several concepts from machine learning and mathematics.

The same ideas appear in:

- Gradient Descent
- Decision Trees
- Regression
- Mean Squared Error
- Loss Functions
- Ensemble Learning
- Function Approximation
- Optimization

The most important connection is that Gradient Boosting applies the idea of **gradient descent to functions**, using decision trees as the individual correction functions.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented in Python.
