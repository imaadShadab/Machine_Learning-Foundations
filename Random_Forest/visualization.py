import matplotlib.pyplot as plt
import numpy as np
from randomForest import RandomForest

# Example dataset
x1 = np.array([
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 8],
    [9, 10],
])

y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])


forest = RandomForest(n_trees=4)
forest.fit(x1, y1)


def draw_tree(ax, node, x_pos, y_pos, dx):

    if node.prediction is not None:
        ax.text(
            x_pos,
            y_pos,
            f"Predict {node.prediction}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", fc="lightgreen"),
        )
        return

    ax.text(
        x_pos,
        y_pos,
        f"x[{node.feature}] < {node.threshold:.2f}",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", fc="lightblue"),
    )

    left_x = x_pos - dx
    right_x = x_pos + dx
    child_y = y_pos - 1

    ax.plot([x_pos, left_x], [y_pos, child_y], "k-")
    ax.plot([x_pos, right_x], [y_pos, child_y], "k-")

    draw_tree(ax, node.left, left_x, child_y, dx / 2)
    draw_tree(ax, node.right, right_x, child_y, dx / 2)


fig, axes = plt.subplots(2, 2)

for i, ax in enumerate(axes.flat):
    ax.set_axis_off()
    ax.set_title(f"Tree {i + 1}")
    draw_tree(ax, forest.trees[i].root, 0, 0, 4)

plt.tight_layout()
plt.show()