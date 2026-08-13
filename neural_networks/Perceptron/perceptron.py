import numpy as np

x1 = np.array([[1, 1], [2, 1], [1, 2], [2, 2], [3, 2], [6, 5], [7, 6], [6, 7], [8, 7], [9, 8]])

y1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


class Perceptron():
    def __init__(self):
        self.x = x1
        self.y = y1
        self.weights = np.zeros(self.x.shape[1])
        self.bias = 0
        self.learning_rate = 0.5
        self.epochs = 100
        
    def fit(self, x, y, epochs = None):
        self.x = x
        self.y = y
        self.epochs = epochs
        for _ in range(epochs):
            mistakes = 0
            for i in range(len(self.x)):
                z = (self.x[i] @ self.weights) + self.bias  # z = score
                prediction = int(z>=0)
                
                if prediction != self.y[i]:
                    error = self.y[i] - prediction
                    self.weights = self.weights + ((self.learning_rate * error) * self.x[i])
                    self.bias = self.bias + (self.learning_rate * (error))
                    mistakes += 1
            if mistakes == 0:
                break
           
    def predict(self, x_new):
        z = (x_new @ self.weights) + self.bias
        prediction = int(z>=0)
        return prediction
        


def main():
    p = Perceptron()
    p.fit(x1, y1, 2000)
    x_new = [5,1]
    predict = p.predict(x_new)
    print(f'{x_new} belongs to class{predict}')
    print(f'weights: {p.weights}')
    print(f'bias: {p.bias}')
    
    

if __name__ == '__main__':
    main()
