import numpy as np
import matplotlib.pyplot as plt

from gradientBoosting import GradientBoosting


# Nonlinear dataset for visualization
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

y = np.array([
    2.1,
    3.8,
    5.2,
    7.1,
    10.5,
    12.2,
    11.8,
    10.1,
    8.3,
    6.2
])


# Train model
gb = GradientBoosting()
gb.fit(X, y)


# Generate points for prediction
X_plot = np.linspace(1, 10, 200).reshape(-1, 1)
y_pred = gb.predict(X_plot)


# Plot
plt.scatter(X, y, label="Training Data")
plt.plot(X_plot, y_pred, label="Gradient Boosting Prediction")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Gradient Boosting Regression")
plt.legend()
plt.show()