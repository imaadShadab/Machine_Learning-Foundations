import matplotlib.pyplot as plt
import numpy as np

from bernoulliNaiveBayes import (calculate_priors, 
                                 calculate_feature_probablities, 
                                 prediction)


X = np.array([
    [1,1,0],
    [1,0,0],
    [1,1,0],
    [0,0,1],
    [0,1,1],
    [0,0,1]
])

y = np.array([1,1,1,0,0,0])

x_new = np.array([1,1,0])

priors = calculate_priors(y)
feature_probs = calculate_feature_probablities(X, y)

scores = []

for cls in sorted(priors.keys()):

    score = np.log(priors[cls])

    for feature, prob in zip(x_new, feature_probs[cls]):

        if feature:
            score += np.log(prob)
        else:
            score += np.log(1 - prob)

    scores.append(score)

classes = [f"Class {c}" for c in sorted(priors.keys())]

plt.bar(classes, scores)

plt.ylabel("Log Probability")

plt.title(f"Prediction for {x_new}")

for i, score in enumerate(scores):
    plt.text(i, score, f"{score:.2f}", ha="center")

plt.show()