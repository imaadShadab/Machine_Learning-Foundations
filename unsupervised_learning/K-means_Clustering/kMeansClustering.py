import numpy as np

x1 = np.array(
    [
        [1, 2],
        [2, 1],
        [2, 3],
        [3, 2],
        
        [8, 8],
        [9, 7],
        [8, 9],
        [9, 9],
        
        [2, 9],
        [3, 8],
        [1, 8],
        [2, 7],
    ]
)


class KMeans:

    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.clusters = None
        self.x = None
        

    def initialize_centroids(self, x):
        indices = np.random.choice(len(x), self.k, replace=False)
        return x[indices]
    
    def calculate_distance(self, data_point, centre):

        distance = np.sqrt(np.sum((data_point - centre) ** 2))
        return distance

    def assign_clusters(self, x):
        clusters = []
        for i in range(self.k):
            clusters.append([])

        for data in x:
            temp_distances = []

            for centre in self.centroids:
                temp_distances.append(self.calculate_distance(data, centre))

            clusters[np.argmin(temp_distances)].append(data)

        return clusters

    def calculate_centroids(self):
      
        centroids = []

        for cluster in self.clusters:

            if len(cluster) == 0:
                random_index = np.random.choice(len(self.x))
                centroids.append(self.x[random_index])
            else:
                centroids.append(np.mean(cluster, axis=0))

        return np.array(centroids)
    
    def fit(self, x):
        
        self.x = x
        self.centroids = self.initialize_centroids(x)
        previous_centroids = None
        iterations = 0

        while ( (not np.array_equal(previous_centroids, self.centroids))
               and 
               (iterations < self.max_iterations) ):
            
            previous_centroids = self.centroids.copy()
            self.clusters = self.assign_clusters(x)
            self.centroids = self.calculate_centroids()

            iterations += 1
            
    
    def predict(self, x_new):
        distances = []
        for centre in self.centroids:
            distances.append(self.calculate_distance(x_new, centre))
        
        return np.argmin(distances)
        
                
        
def main():
    k = KMeans(3)
    k.fit(x1)

    x_new = (10,10)
    predict = k.predict(x_new)

    print(f"{x_new} belongs to cluster {predict}")

if __name__ == "__main__":
    main()