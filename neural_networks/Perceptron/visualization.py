import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

x = np.array(
    [[1, 1], [2, 1], [1, 2], [2, 2], [3, 2], [6, 5], [7, 6], [6, 7], [8, 7], [9, 8]]
)

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


p = Perceptron()
p.fit(x, y, 100)

class_0 = x[y == 0]
class_1 = x[y == 1]


plt.scatter(class_0[:, 0], class_0[:, 1], label="Class 0")

plt.scatter(class_1[:, 0], class_1[:, 1], label="Class 1")


w1 = p.weights[0]
w2 = p.weights[1]
b = p.bias


x_boundary = np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 100)

y_boundary = -(w1 * x_boundary + b) / w2


plt.plot(x_boundary, y_boundary, label="Decision Boundary")


plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid()

plt.show()
