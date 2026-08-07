import matplotlib.pyplot as plt
import numpy as np
from supervised_learning.Locally_Weighted_Regression.localWeightRegression import locally_weighted_regression

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.2, 1.8, 3.0, 4.5, 6.8, 6.5, 5.0, 4.0, 3.8, 4.2])

design_matrix_1 = np.column_stack((np.ones(x.shape), x))
theta_params = np.zeros(design_matrix_1.shape[1])
target_x = 4.5
bandwidth_1 = 2


# Query points where we'll evaluate the model
query_points = np.linspace(min(x), max(x), 500)
predictions = []
for target in query_points:

    prediction = locally_weighted_regression(design_matrix_1, target, bandwidth_1)

    predictions.append(prediction)

target_prediction  = locally_weighted_regression(design_matrix_1, target_x, bandwidth_1)

# Plot

plt.scatter(x, y, color="blue", label="Training Data")
plt.plot(query_points, predictions, color="red", linewidth=2, label="Locally Weighted Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Locally Weighted Regression")
plt.legend()
plt.grid(True)
plt.scatter(
    target_x,
    target_prediction,
    color="green",
    s=100,
    marker="x",
    label="Prediction at x=4.5"
)
plt.show()