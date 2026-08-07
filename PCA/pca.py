import numpy as np

x1 = np.array([
    [2.0, 1.0],
    [3.0, 2.0],
    [4.0, 3.0],
    [5.0, 4.0],
    [6.0, 5.0],
    [7.0, 6.0],
    [8.0, 7.0],
    [9.0, 8.0],
])

class PCA:

    def __init__(self, k):
        self.k = k      # Here, k equals how many top eigenvalues sorted by size we want to take 
        self.x = None
        self.P_matrix = None    # P_matrix = Projection Matrix
    
    def centre_data(self):
        
        for feature_index in range(self.x.shape[1]):
            
            feature = self.x[:, feature_index]
            mean = np.mean(feature)
            
            for i in range(len(feature)):
                feature[i] = feature[i] - mean
                
    def compute_covariance(self):
        '''rowvar means that rows are treated as variables(Features),
        however our dataset (and most other datasets) treat columns as features,
        therefore we set rowvar = False'''
        covariance_matrix = np.cov(self.x, rowvar=False)
        return covariance_matrix
            
    def compute_eigens(self, cov_matrix):
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        sorted_indices = np.argsort(eigenvalues)
        sorted_indices = sorted_indices[::-1]
        
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        return eigenvectors
    
    def projection_matrix(self, eigenvectors):
        projection_matrix = eigenvectors[:, :self.k]
        return projection_matrix
    
    def fit(self, x):
        self.x = x
        self.centre_data()
        cov_matrix = self.compute_covariance()
        eigenvectors = self.compute_eigens(cov_matrix) 
        proj_matrix = self.projection_matrix(eigenvectors) 
        
        self.P_matrix = proj_matrix
    
    def transform(self, X):
        x_new = X @ self.P_matrix
        return x_new
        
        
pca = PCA(2)

print(pca.projection_matrix())