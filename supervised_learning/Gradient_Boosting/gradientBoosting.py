import numpy as np
from sklearn.tree import DecisionTreeRegressor

x1 = np.array(
    [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
    ]
)

y1 = np.array([3, 5, 7, 9, 11, 13, 15, 17])


class GradientBoosting:

    def __init__(self):
        self.x = None
        self.y = None
        self.mean = None
        self.score = None
        self.prediction = None
        self.max_trees = 20
        self.learning_rate = 0.5
        self.trees = []

    def calculate_residuals(self):
        return self.y - self.prediction
    
    def train_tree(self):
        
        residuals = self.calculate_residuals()
        tree = DecisionTreeRegressor(max_depth=2)
        tree.fit(self.x, residuals)
        predict = tree.predict(self.x) 
        self.prediction += self.learning_rate * predict
        return tree
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        
        # Initializing the model by calculating mean of y as first predictions
        self.prediction = np.mean(self.y)
        self.mean = self.prediction
        for _ in range(self.max_trees):
            tree = self.train_tree()
            self.trees.append(tree)
    
    def predict(self, x_new):
        prediction = np.full(len(x_new), self.mean)

        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(x_new)

        return prediction
        

gb = GradientBoosting()
gb.fit(x1, y1)
import numpy as np
from sklearn.tree import DecisionTreeRegressor

x1 = np.array(
    [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
    ]
)

y1 = np.array([3, 5, 7, 9, 11, 13, 15, 17])


class GradientBoosting:

    def __init__(self):
        self.x = None
        self.y = None
        self.mean = None
        self.score = None
        self.prediction = None
        self.max_trees = 20
        self.learning_rate = 0.5
        self.trees = []

    def calculate_residuals(self):
        return self.y - self.prediction
    
    def train_tree(self):
        
        residuals = self.calculate_residuals()
        tree = DecisionTreeRegressor(max_depth=2)
        tree.fit(self.x, residuals)
        predict = tree.predict(self.x) 
        self.prediction += self.learning_rate * predict
        return tree
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        
        # Initializing the model by calculating mean of y as first predictions
        self.prediction = np.mean(self.y)
        self.mean = self.prediction
        for _ in range(self.max_trees):
            tree = self.train_tree()
            self.trees.append(tree)
    
    def predict(self, x_new):
        prediction = np.full(len(x_new), self.mean)

        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(x_new)

        return prediction
        

gb = GradientBoosting()
gb.fit(x1, y1)
print(gb.predict(np.array([[9]])))