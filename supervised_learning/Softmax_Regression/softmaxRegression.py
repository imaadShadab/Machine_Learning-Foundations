import numpy as np

x1 = np.array([
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

y1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) # K (All classes, here K=3; {0,1,2})

learning_alpha = 0.01

def softmax(z):
    scores_exp =  np.exp(z)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    probablities = scores_exp / scores_exp_sum
    return probablities
    

def softmax_regression(x, y, learning_rate, max_iterations):

    # Number of training examples
    m = len(y)
    design_matrix = np.column_stack((np.ones(x.shape[0]), x))

    # How many classes in y
    K_classes = len(set(y))

    # design_matrix.shape[1] gives us how many columns in it;
    # len(set(y)) gives us # of unique classes(K) in y
    theta_matrix = np.zeros((design_matrix.shape[1], K_classes))
    
    Y_outputMatrix = np.eye(K_classes)[y]
    
    last_cost = float("inf")

    for i in range(max_iterations):
        scores = design_matrix @ theta_matrix

        probablities = softmax(scores)

        residual_matrix = probablities - Y_outputMatrix

        gradient_matrix = (design_matrix.T @ residual_matrix) / m

        theta_matrix = theta_matrix - (learning_rate * gradient_matrix)

        cost = (-1 / m) * np.sum(np.sum(Y_outputMatrix * np.log(probablities)))

        if abs(last_cost - cost) < 1e-5:
            break

        last_cost = cost

        # print(f'iter{i}- P: {probablities}')
        # print(f'iter{i}- Error_matrix: {residual_matrix}')
        # print(f'iter{i}- gradient_matrix: {gradient_matrix}')
        # print(f'iter{i}- theta_matrix: {theta_matrix}')
        # print(f'iter{i}- Cost: {cost}')
        # print("_____________________________________________________")

    return theta_matrix

def predict(x, theta_matrix):

    design_matrix = np.column_stack((np.ones(x.shape[0]), x))

    scores = design_matrix @ theta_matrix

    probabilities = softmax(scores)

    predictions = np.argmax(probabilities, axis=1)

    return predictions