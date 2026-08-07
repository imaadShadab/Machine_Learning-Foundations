import numpy as np
import matplotlib.pyplot as plt

from supervised_learning.Logistic_Regression.logisticRegression import logistic_regression, sigmoid


x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

design_matrix = np.column_stack((np.ones(x.shape), x))

theta = logistic_regression(
    design_matrix,
    y,
    learning_rate=0.01,
    max_iteration=50000
)

x_plot = np.linspace(x.min(), x.max(), 300)
X_plot = np.column_stack((np.ones(x_plot.shape), x_plot))

probabilities = sigmoid(X_plot @ theta)


x_value = 4.5
X_value = np.array([1, x_value])
y_value = sigmoid(X_value @ theta)

# Plot the point
plt.scatter(
    x_value,
    y_value,
    color="purple",
    s=100,
    zorder=5,
    label=f"x={x_value}, p={y_value:.2f}"
)
plt.scatter(x, y, color="red", label="Training Data")
plt.plot(x_plot, probabilities, label="Sigmoid Curve")
plt.axhline(0.5, linestyle="--", color="gray")

decision_boundary = -theta[0] / theta[1]
plt.axvline(decision_boundary, linestyle="--", color="green",
            label="Decision Boundary")

plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.legend()
plt.show()