import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


# -----------------------------
# Dataset
# -----------------------------

x = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [1, 3],
    [3, 2],
    [2, 3]
])

y = np.array([
    4,
    6,
    7,
    9,
    12,
    8,
    14,
    11
])


# -----------------------------
# Train network
# -----------------------------

nn = NeuralNetwork(3)
nn.fit(x, y, 700)

predictions = nn.predict(x).flatten()


# -----------------------------
# Visualize network architecture
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Input → Hidden connections
for input_y in [2, 1]:
    for hidden_y in [3, 2, 1]:
        ax.plot(
            [0, 1],
            [input_y, hidden_y],
            linewidth=1
        )

# Hidden → Output connections
for hidden_y in [3, 2, 1]:
    ax.plot(
        [1, 2],
        [hidden_y, 2],
        linewidth=1
    )

# Input neurons
for y_pos in [2, 1]:
    ax.scatter(0, y_pos, s=1000, zorder=3)

# Hidden neurons
for y_pos in [3, 2, 1]:
    ax.scatter(1, y_pos, s=1000, zorder=3)

# Output neuron
ax.scatter(2, 2, s=1000, zorder=3)

# Labels
ax.text(0, 0.5, "Input\n2 Features", ha="center", fontsize=12)
ax.text(1, 0.5, "Hidden Layer\n3 Neurons\nReLU", ha="center", fontsize=12)
ax.text(2, 0.5, "Output\n1 Neuron\nLinear", ha="center", fontsize=12)

ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 3.5)
ax.set_title("Neural Network Architecture")
ax.axis("off")

plt.show()


# -----------------------------
# Actual vs Predicted
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    range(len(y)),
    y,
    label="Actual",
    s=60
)

plt.scatter(
    range(len(predictions)),
    predictions,
    label="Predicted",
    marker="x",
    s=70
)

plt.xlabel("Sample")
plt.ylabel("Target (y)")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(True)

plt.show()