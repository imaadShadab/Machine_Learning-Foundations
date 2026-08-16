import numpy as np
import math_functions as mf

x1 = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y1 = np.array([
     3,
     5,
     7,
     9,
    11,
    13,
    15,
    17
])


class RegressionNeuralNetwork:
    def __init__(self):
        self.x = None
        self.y = None
        self.epochs = 100
        self.weights = None
        self.bias = 0
        self.learning_rate = 0.001
        
    
    def forward_propagation(self):
        '''Since this is a regression problem out activation function is a = f(z) = z,
        which gives us a = z. Therefore we wont explicitly write "a". '''
        z  = (self.x @ self.weights) + self.bias
        return z

    def calculate_loss(self, prediction):
        return mf.mse_loss(prediction, self.y)

    def back_propagation(self, prediction):
        
        dW = mf.weight_gradient(
            self.x.flatten(),
            prediction,
            self.y
        )

        db = mf.bias_gradient(
            prediction,
            self.y
        )

        return dW, db
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.weights = np.zeros(self.x.shape[1])
        for epoch in range(self.epochs):

            # Forward
            prediction = self.forward_propagation()

            # Loss
            loss = self.calculate_loss(prediction)

            # Backward
            dW, db = self.back_propagation(prediction)

            # Update
            self.weights -= self.learning_rate * np.mean(dW)
            self.bias -= self.learning_rate * np.mean(db)

            print(f"Epoch {epoch}: loss = {loss}")
        
        
        
    
rnn = RegressionNeuralNetwork()

rnn.fit(x1, y1)

print("weights:", rnn.weights)
print("bias:", rnn.bias)