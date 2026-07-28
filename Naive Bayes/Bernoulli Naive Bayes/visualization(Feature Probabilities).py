import matplotlib.pyplot as plt
import numpy as np

from bernoulliNaiveBayes import calculate_priors, calculate_feature_probablities, prediction


X = np.array([
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 0, 1]
])

y = np.array([1, 1, 1, 0, 0, 0])

feature_names = ["Free", "Money", "Meeting"]

priors = calculate_priors(y)
feature_probs = calculate_feature_probablities(X, y)

classes = sorted(feature_probs.keys())

fig, axes = plt.subplots(1, len(classes), figsize=(10, 4), sharey=True)

if len(classes) == 1:
    axes = [axes]

for ax, cls in zip(axes, classes):

    probs = feature_probs[cls]

    ax.bar(feature_names, probs)

    ax.set_ylim(0, 1)

    ax.set_title(f"Class {cls}")

    ax.set_ylabel(r"$P(feature=1 \mid class)$")

    for i, p in enumerate(probs):
        ax.text(
            i,
            p + 0.03,
            f"{p:.2f}",
            ha="center"
        )

plt.suptitle("Bernoulli Naive Bayes Feature Probabilities")
plt.tight_layout()
plt.show()