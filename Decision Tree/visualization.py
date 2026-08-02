import matplotlib.pyplot as plt
import numpy as np

from decisionTree import build_tree

# x1 = np.array(
#     [
#         [2, 2],
#         [2, 8],
#         [3, 3],
#         [3, 7],
#         [4, 2],
#         [4, 8],
#         [6, 2],
#         [6, 8],
#         [7, 3],
     
#     ]
# )

# y1 = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1])


x1 = np.array([
    [2, 3], 
    [3, 4], 
    [4, 3], 
    [5, 6], 
    [6, 7], 
    [7, 8], 
    [8, 8], 
    [9, 10],
    [11, 12],
    [12, 15], 
    [17, 18], 
    [8, 9]
])

# Lets assume class0 = cat, class1 = Dog
y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

root = build_tree(x1, y1)


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_axis_off()


def draw_tree(node, x_pos, y_pos, dx):

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

    draw_tree(node.left, left_x, child_y, dx / 2)
    draw_tree(node.right, right_x, child_y, dx / 2)


draw_tree(root, 0, 0, 4)

plt.title("Decision Tree")
plt.show()
