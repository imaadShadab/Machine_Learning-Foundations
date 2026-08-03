import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.2, 1.8, 3.0, 4.5, 6.8, 6.5, 5.0, 4.0, 3.8, 4.2])

design_matrix_1 = np.column_stack((np.ones(x.shape[0]), x))
theta_params = np.zeros(design_matrix_1.shape[1])
target_x = 4.5
bandwidth_1 = 2


def locally_weighted_regression(design_matrix, target, bandwidth):
    weights = []
    for i in x:
        distance = target - i
        w = np.exp(-((distance)**2) / (2*(bandwidth)**2)) # refer equation 4 in notes
        weights.append(w)

    diagonal_weight_matrix = np.diag(weights)

    # Normal Equation
    XtWy = design_matrix.T @ diagonal_weight_matrix @ y
    XtWX = np.linalg.inv(design_matrix.T @ diagonal_weight_matrix @ design_matrix)
    solution = XtWX @ XtWy 

    target_x_designMatrix = np.array([1, target]) # To multiply matrices

    hypothesis = target_x_designMatrix @ solution
    return hypothesis



prediction = locally_weighted_regression(design_matrix_1, target_x, bandwidth_1)
print(prediction)