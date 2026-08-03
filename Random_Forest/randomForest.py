'''RUN THIS COMMAND IN SHELL TO RUN THIS FILE    
python -m Random_Forest.randomForest'''

import numpy as np

from Decision_Tree.decisionTree_OOP import DecisionTree

# ex: Height and Weight of an Animal
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

# Lets assume class0 = cat, class1 = Dog
y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])



class RandomForest:
    
    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self.trees = []
    
    def bootstrap_sampling(self, x, y):
        
        sample_size = len(x)        
        random_indices = np.random.choice(sample_size, sample_size, replace=True)
        
        x_bootstrap = x[random_indices]
        y_bootstrap = y[random_indices]
        
        return x_bootstrap, y_bootstrap
      
    '''Moved to decisionTree_OOP as it being there simplifies the design and makes sense logically
    since the information_gain and data_splitting is a part of decisionTree_OOP'''  
    # # def random_feature_selection(self, x):
    # #     column_count = x.shape[1]
    # #     random_feature_count = round(np.sqrt(column_count))
    # #     random_feature_indices = np.random.choice(column_count, random_feature_count, replace=False)
    # #     random_features = x[:, random_feature_indices]
        
        
    #     '''We return the indices rather than feature so later in information_gain we can easily map
    #     the indices and iterate ober them'''
    #     return random_feature_indices
    
    def fit(self, x, y):
        for _ in range(self.n_trees):
            boostrap_x, bootstrap_y = self.bootstrap_sampling(x, y)

            
            tree = DecisionTree()
            tree.fit(boostrap_x, bootstrap_y,random_forest=True)
            self.trees.append(tree)
    
    def predict(self, x_new):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(x_new))
        
        values, counts = np.unique(predictions, return_counts=True)
        max_index = np.argmax(counts)
        majority = values[max_index]
        print(values, counts)
        return majority
    
        
        
            


forest1 = RandomForest(n_trees=1000)
forest1.fit(x1, y1)
x = forest1.predict([5,1])
print(x)

