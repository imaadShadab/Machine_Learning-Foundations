import matplotlib.pyplot as plt
import numpy as np
from linearRegression import linear_regression

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 5, 7, 8, 11])

design_matrix = np.column_stack((np.ones(x.shape), x))

theta = linear_regression(
    design_matrix,
    y,
    learning_rate=0.01,
    max_iteration=5000
)

predictions = design_matrix @ theta


# Training data
plt.scatter(x, y, color="blue", label="Training Data")

# Regression line
plt.plot(x, predictions, color = "red", linewidth=2, label="Linear Regression")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)

plt.show()