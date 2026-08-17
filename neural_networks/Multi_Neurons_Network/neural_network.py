import numpy as np

x1 = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 1],
    [1, 3],
    [3, 2],
    [2, 3]
])

y1 = np.array([
     4,
     6,
    7,
    9,
    12,
    8,
    14,
    11
])

class NeuralNetwork:
    def __init__(self, neurons):
        self.x = None
        self.y = None
        self.neurons = neurons
        self.weights = None
        self.bias = np.zeros(self.neurons)
        self.output_weight = np.random.randn(1, self.neurons) * 0.1
        self.output_bias = np.zeros(1)
        self.learning_rate = 0.01
        self.epoch = 100
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward_propogation(self, x):
        z = (x @ self.weights.T) + self.bias
        a = self.relu(z)
        
        # Output layer
        prediction = (a @ self.output_weight.T) + self.output_bias
        # No activation function here since the problem is regression
        return prediction, a, z
    
    def calculate_cost(self, prediction):
        return np.mean(1/2 * (prediction - self.y)**2)
    
    def back_propogation(self):
        
        prediction, a, z = self.forward_propogation(self.x)
        # 1st we calculate dC/dy` (or) how does changing prediction changes the cost
        output_gradient = (prediction - self.y) / len(self.y)
        
        """
        2nd we calculate dC/dw (or) how does changing weight changes the cost
        Uses the chain rule:    
        dC/dw = dC/dŷ * dŷ/dz * dz/dw
        For our linear output:
        dC/dŷ = prediction - y
        dŷ/dz = 1
        dz/dw = a
        """
        # output_weight_gradient = output_gradient * 1 * a
        
        # Reshaping the output weight gradient to match the actual shape (1,3)
        output_weight_gradient = np.sum(output_gradient * a, axis=0).reshape(1, -1)
        
        """
        Calculating the bias gradient 
        dC/db
        
        dC/db = dC/dŷ * dŷ/dz * dz/db
        """
        # output_bias_gradient = output_gradient * 1 * 1
        output_bias_gradient = np.sum( output_gradient, axis=0)
                
        """
        Now we calculate dC/dA (or) how does changing the hidden-layer
        activation changes the cost

        Uses the chain rule:
        dC/dA = dC/dŷ * dŷ/dz * dz/dA

        For our linear output:
        dC/dŷ = prediction - y
        dŷ/dz = 1
        dz/dA = output_weight

        The output weight determines how strongly each hidden
        activation affects the prediction, so it determines how
        much gradient is passed back to each hidden neuron.
        """
        activation_gradient = output_gradient @ self.output_weight
        
        """
        Then we calculate dC/dz (or) how does changing the hidden-layer
        z affect the cost

        Uses the chain rule:
        dC/dz = dC/dA * dA/dz

        For our ReLU activation:
        dC/dA = activation_gradient
        dA/dz = ReLU'(z)

        ReLU'(z) = 1 when z > 0
        ReLU'(z) = 0 when z <= 0

        ReLU acts like a gate during backpropagation:
        if z > 0, the gradient passes through
        if z <= 0, the gradient becomes 0.
        """
        relu_gradient = (z > 0).astype(int)

        z_gradient = activation_gradient * relu_gradient
        
        """
        we calculate dC/dW_hidden (or) how does changing a hidden-layer
        weight change the cost.

        Uses the chain rule:

        dC/dW = dC/dZ * dZ/dW

        We already calculated:
        dC/dZ = z_gradient

        For a hidden neuron:
        z = x₁w₁ + x₂w₂ + b

        Therefore:
        dz/dw = x

        So:
        dC/dW = z_gradient * X
        """
        hidden_neurons_weight_gradient = z_gradient.T @ self.x
            
        """
        Finally, we calculate dC/db_hidden (or) how does changing a hidden-layer
        bias change the cost.

        Uses the chain rule:
        dC/db = dC/dZ * dZ/db

        We already calculated:
        dC/dZ = z_gradient

        Since:
        z = XW + b

        the derivative of z with respect to b is:
        dz/db = 1

        Therefore:
        dC/db = z_gradient * 1
        """
        hidden_neurons_bias_gradient = np.sum(z_gradient, axis=0)
        
        return (hidden_neurons_weight_gradient,
                hidden_neurons_bias_gradient,
                output_weight_gradient,
                output_bias_gradient)
        
    def gradient_descent(self):
        '''This function updates the appropirate weights and biases'''
        (hidden_neurons_weight_gradient,
        hidden_neurons_bias_gradient,
        output_weight_gradient,
        output_bias_gradient) = self.back_propogation()
        
        self.weights -= self.learning_rate * hidden_neurons_weight_gradient
        self.bias -= self.learning_rate * hidden_neurons_bias_gradient

        self.output_weight -= self.learning_rate * output_weight_gradient
        self.output_bias -= self.learning_rate * output_bias_gradient
        
    def fit(self, x, y, epoch = None):
        self.x = x
        self.y = y.reshape(-1, 1)
        self.weights = np.random.randn(self.neurons, self.x.shape[1]) * 0.1
        
        if epoch is None:
            epoch = self.epoch
            
        for _ in range(epoch):
            prediction, a, z = self.forward_propogation(self.x)
            cost = self.calculate_cost(prediction)
            self.gradient_descent()
            # print(f"Epoch {_}: cost = {cost}")
            
    def predict(self, x_new):
        prediction, a, z = self.forward_propogation(x_new)
        return prediction
    
            
def main():
    nn = NeuralNetwork(3)
    nn.fit(x1,y1, 700)
    x_new = np.array([[1, 1]])
    print(nn.predict(x_new))

if __name__ == "__main__":
    main()