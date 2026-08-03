# Bernoulli Naive Bayes from Scratch

This project implements **Bernoulli Naive Bayes** from scratch using **NumPy**.

Bernoulli Naive Bayes is a probabilistic machine learning algorithm used for **binary classification** when the input features are binary (0 or 1). Instead of learning a decision boundary like Logistic Regression, it models the probability of each class and uses **Bayes' Theorem** to determine which class is most likely to have generated the observed data.

The objective was to understand probabilistic classification, Bayes' Theorem, Laplace Smoothing, and log probabilities before implementing the algorithm.

---

# Objective

Given a dataset

| Free | Money | Meeting | Class |
|------|-------|----------|-------|
| 1 | 1 | 0 | Spam |
| 1 | 0 | 0 | Spam |
| 1 | 1 | 0 | Spam |
| 0 | 0 | 1 | Ham |
| 0 | 1 | 1 | Ham |
| 0 | 0 | 1 | Ham |

learn the probability distribution of every feature for every class, then classify a new email by selecting the class with the highest posterior probability.

---

# Mathematical Background

Bernoulli Naive Bayes predicts the most likely class using **Bayes' Theorem**.

Instead of directly learning a decision boundary, it computes the probability that an example belongs to each possible class.

---

## Prior Probability

The prior represents how common each class is in the training data.

```text
P(y = k)
```

It is computed as

```text
P(y = k) =
(Number of samples in class k)
--------------------------------
(Total number of samples)
```

---

## Bayes' Theorem

For a new observation

```text
x = (x₁, x₂, ..., xₙ)
```

the posterior probability is

```text
          P(x | y) P(y)
P(y | x) = ---------------
             P(x)
```

Since every class shares the same denominator, we only compare

```text
P(x | y) P(y)
```

---

## Naive Assumption

Naive Bayes assumes that every feature is conditionally independent given the class.

This allows the likelihood to be written as

```text
P(x | y)

=

P(x₁ | y)
P(x₂ | y)
...
P(xₙ | y)
```

Although this assumption is often unrealistic, the algorithm performs surprisingly well on many classification problems.

---

## Bernoulli Likelihood

Because the features are binary, each feature follows a Bernoulli distribution.

The likelihood of a single feature is

```text
P(xᵢ | y)

=

pᵢˣⁱ (1-pᵢ)¹⁻ˣⁱ
```

where

- **pᵢ** is the probability that feature **i** equals 1.
- **xᵢ** is either 0 or 1.

This compact formula automatically handles both cases.

---

## Laplace Smoothing

Without smoothing, unseen features would receive probability zero.

To avoid this, Laplace Smoothing is applied.

```text
          Count + 1
P = ---------------------
    Class Count + 2
```

This guarantees that every feature probability is strictly between 0 and 1.

---

## Log Probabilities

Multiplying many probabilities quickly produces extremely small numbers.

Instead of computing

```text
P(y)

×

P(x₁|y)

×

P(x₂|y)

×

...
```

the implementation computes

```text
log(P(y))

+

log(P(x₁|y))

+

log(P(x₂|y))

+

...
```

Since the logarithm is monotonic, the class with the largest probability also has the largest log probability.

---

# Algorithm

1. Compute the prior probability for every class.
2. Count how often every feature equals 1 within each class.
3. Apply Laplace Smoothing.
4. Store the feature probability vectors.
5. For a new sample, compute the log probability for every class.
6. Select the class with the highest score.

---

# Implementation Notes

The implementation is generalized to support **any number of classes**.

During training the model computes

```python
priors = calculate_priors(y)

feature_probabilities = calculate_feature_probabilities(X, y)
```

During prediction

```python
score = log(prior)

for each feature:

    if feature == 1:
        score += log(probability)

    else:
        score += log(1 - probability)
```

The predicted class is the one with the largest log score.

---

# Visualization

The learned feature probabilities are visualized using Matplotlib.

Each bar represents

```text
P(feature = 1 | class)
```

for every feature.

The visualization provides an intuitive view of which words are most strongly associated with each class.

The prediction process can also be visualized by comparing the final log probability of every class.

---

# Files

```text
bernoulli_naive_bayes/
│
├── naive_bayes.py
├── visualization.py
└── README.md
```

---

# Concepts Learned

- Bayes' Theorem
- Prior Probability
- Posterior Probability
- Conditional Probability
- Bernoulli Distribution
- Naive Independence Assumption
- Laplace Smoothing
- Log Probabilities
- Binary Classification
- Vectorization

---

# Future Improvements

- [ ] Multinomial Naive Bayes
- [ ] Gaussian Naive Bayes
- [ ] Categorical Naive Bayes
- [ ] Prediction Explanation Utility
- [ ] Confusion Matrix
- [ ] Accuracy, Precision, Recall and F1 Score
- [ ] Feature Importance Visualization
- [ ] Support Continuous Features

---

# Connections

While implementing this algorithm, I found that Bernoulli Naive Bayes approaches classification differently from Logistic Regression.

Instead of learning a decision boundary through optimization, Naive Bayes models the probability distribution of each class and applies Bayes' Theorem to determine the most probable class for a new observation.

The same mathematical ideas appear in several topics:

- Probability Theory
- Conditional Probability
- Bayes' Theorem
- Bernoulli Distribution
- Laplace Smoothing
- Maximum Likelihood Estimation
- Generative Models
- Information Theory (Log Probabilities)

These concepts form the mathematical foundation for many probabilistic machine learning algorithms and Bayesian methods.

---

This implementation is part of the **Machine Learning Foundations** repository, where each algorithm is studied mathematically before being implemented from scratch in Python.