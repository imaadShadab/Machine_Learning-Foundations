# Decision Tree from Scratch

This project implements a **Decision Tree Classifier** from scratch using **NumPy** and **recursive tree construction**.

Unlike Linear and Logistic Regression, a Decision Tree does not learn a set of numerical parameters. Instead, it recursively partitions the dataset by selecting the feature and threshold that maximize **Information Gain**, building a tree of decision nodes and prediction (leaf) nodes.

The objective was to understand how a Decision Tree learns decision boundaries before implementing the complete training algorithm from scratch.

---

# Objective

Given a dataset

| Height | Weight | Class |
|--------:|--------:|------:|
| 2 | 3 | 0 |
| 3 | 4 | 0 |
| 4 | 3 | 0 |
| 5 | 6 | 0 |
| 6 | 7 | 1 |
| 7 | 8 | 1 |
| 8 | 8 | 1 |
| 9 | 10 | 1 |

learn a sequence of decision rules that correctly classify new samples.

---

# Mathematical Background

Decision Trees recursively divide the dataset into smaller subsets.

At every node, every feature and every candidate threshold are evaluated.

The split that produces the highest **Information Gain** becomes the decision rule stored in that node.

---

## Entropy

Entropy measures the impurity of a dataset.

```text
H(S) = -Σ pᵢ log₂(pᵢ)
```

where

- **pᵢ** is the probability of class *i*.

Low entropy indicates that most samples belong to the same class.

---

## Candidate Thresholds

For each feature, candidate thresholds are generated using the midpoints between consecutive unique feature values.

Example

```text
Feature values

2 3 4 6 8

↓

Candidate thresholds

2.5
3.5
5
7
```

Each threshold represents a possible question.

---

## Information Gain

For every candidate threshold,

```text
Information Gain

=

Parent Entropy

−

Weighted Child Entropy
```

The split producing the highest Information Gain is selected.

---

## Recursive Tree Construction

Each recursive call builds one subtree.

The algorithm

1. Checks whether a stopping condition has been reached.
2. Finds the best feature and threshold.
3. Splits the dataset.
4. Recursively builds the left subtree.
5. Recursively builds the right subtree.
6. Returns a decision node.

Eventually the recursion reaches leaf nodes containing class predictions.

---

## Prediction

Prediction starts at the root node.

At every decision node,

```text
Is x[feature] < threshold?
```

If true,

- move to the left child.

Otherwise,

- move to the right child.

The process continues until a leaf node is reached.

The prediction stored in the leaf is returned.

---

# Algorithm

1. Compute the entropy of the current dataset.
2. Evaluate every feature.
3. Generate candidate thresholds.
4. Compute Information Gain for every threshold.
5. Select the best split.
6. Partition the dataset.
7. Recursively build the left subtree.
8. Recursively build the right subtree.
9. Stop when a stopping condition is met.

---

# Stopping Conditions

Tree construction stops when

- all samples belong to one class,
- no useful split can be found,
- the maximum tree depth is reached,
- the number of samples falls below the minimum allowed.

Leaf nodes predict the majority class of the remaining samples whenever the data is not completely pure.

---

# Implementation Notes

The implementation is built entirely from scratch using NumPy.

Main components include

- Entropy calculation
- Information Gain
- Dataset splitting
- Recursive tree construction
- Decision and Leaf nodes
- Tree traversal for prediction

The tree is represented using a custom `Node` class.

---

# Visualization

The repository includes a visualization that recursively draws the learned Decision Tree, displaying

- decision nodes
- split thresholds
- feature indices
- prediction nodes

This provides an intuitive view of the learned model structure.

---

# Files

```text
decision_tree/
│
├── decision_tree.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Decision Trees
- Entropy
- Information Gain
- Recursive Algorithms
- Binary Trees
- Dataset Partitioning
- Tree Traversal
- Greedy Algorithms
- Stopping Criteria
- Binary Classification

---

# Future Improvements

- [ ] Gini Impurity
- [ ] Cost Complexity Pruning
- [ ] Multi-class Classification
- [ ] Random Forest
- [ ] Gradient Boosted Trees
- [ ] Feature Importance
- [ ] Tree Export (Graphviz)

---

# Connections

While implementing this algorithm, I found that Decision Trees combine ideas from several areas of computer science and machine learning.

The same concepts appear in

- Binary Trees
- Recursion
- Divide and Conquer
- Information Theory
- Greedy Algorithms
- Random Forests
- Gradient Boosted Trees

These ideas form the foundation of many modern ensemble learning methods.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.