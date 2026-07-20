import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

learning_rate_alpha = 0.01

design_ones = np.ones(x.shape)

design_matrix_1 = np.column_stack((design_ones, x))

# predictions = design_matrix @ theta_params
 
def linear_regression(design_matrix, target_vector, learning_rate, max_iteration):
    
    theta_params = np.zeros(design_matrix.shape[1])
    last_cost = float("inf")  
    
    for i in range(max_iteration):
        
        # Multiply the design_matrix with theta parameters and substract from currect answer(y) to learn the difference
        residual_vector = ((design_matrix @ theta_params) - target_vector) #Xθ - y (refer BGD in notes)
        
        cost_J_theta = np.sum(residual_vector ** 2) / 2
        
        if abs(last_cost - cost_J_theta) < 1e-20:
            break
        
        # Measures how much each theta param should change to move towards correct answer
        gradient_vector = design_matrix.T@residual_vector # Xθ . x_i^(j) (refer BGD in notes)
        
        theta_params = theta_params - (learning_rate * gradient_vector) # refer equation 3 in notes

        last_cost = cost_J_theta
    #     print(f'iter{i}- Error_vector: {residual_vector}')
    #     print(f'iter{i}- gradient_vector: {gradient_vector}')
    #     print(f'iter{i}- theta_params: {theta_params}')
    #     print(f'iter{i}- Cost: {cost_J_theta}')
    #     print("_____________________________________________________")
        
    # print("Final theta:", theta_params)
    return theta_params
    
    
prediction = linear_regression(design_matrix_1, y, learning_rate_alpha,5000 )
print(prediction)