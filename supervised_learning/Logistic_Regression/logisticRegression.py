import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12])
y_1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0,0,0,0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

learning_rate_alpha = 0.01
design_matrix_1 = np.column_stack((np.ones(x.shape[0]), x))



def logistic_regression(design_matrix, y, learning_rate, max_iteration):
    theta_params = np.zeros(design_matrix.shape[1])
    last_cost = float("inf")  
    for i in range(max_iteration):
        hypothesis = design_matrix @ theta_params
        sigmoid_hypothesis = sigmoid(hypothesis)
        residual_vector = sigmoid_hypothesis - y

        gradient = (design_matrix.T @ residual_vector) / len(y)
        theta_params = theta_params - (learning_rate * gradient)
        
        cost = -(1 / len(y)) * np.sum(y * np.log(sigmoid_hypothesis) + (1 - y) * np.log(1 - sigmoid_hypothesis))
        
        if abs(last_cost - cost) < 1e-5:
            break
        
        last_cost = cost
        
        
        # print(f'iter{i}- Error_vector: {residual_vector}')
        # print(f'iter{i}- gradient_vector: {gradient}')
        # print(f'iter{i}- theta_params: {theta_params}')
        # print(f'iter{i}- Cost: {cost}')
        # print("_____________________________________________________")
   
    # probabilities = sigmoid(design_matrix @ theta_params)
    # # predictions = (probabilities >= 0.5).astype(int)   
    target = np.array([1,9])
    print(sigmoid(target@theta_params))
    
    
    return theta_params

z = logistic_regression(design_matrix_1, y_1, learning_rate_alpha, 4000)
print(z)

