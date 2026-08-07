import numpy as np
import matplotlib.pyplot as plt

from supervised_learning.Softmax_Regression.softmaxRegression import softmax_regression, predict


x = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [5, 5],
    [5, 6],
    [6, 5],
    [9, 1],
    [9, 2],
    [8, 1]
])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


theta_matrix = softmax_regression(
    x,
    y,
    learning_rate=0.01,
    max_iterations=5000
)


x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 300),
    np.linspace(x2_min, x2_max, 300)
)


grid = np.column_stack((
    xx1.ravel(),
    xx2.ravel()
))


predictions = predict(grid, theta_matrix)

predictions = predictions.reshape(xx1.shape)


plt.contourf(
    xx1,
    xx2,
    predictions,
    alpha=0.3
)

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=y,
    edgecolors="black",
    s=80
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Softmax Regression Decision Regions")

plt.show()