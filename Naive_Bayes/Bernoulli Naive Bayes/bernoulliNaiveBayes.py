import numpy as np

x1 = np.array([
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 0, 1]
])

y1 = np.array([1, 1, 1, 0, 0, 0])

'''Calculate prior for all classes K'''
def calculate_priors(y):
    y_len = len(y)
    priors = {}
    unique_classes = np.unique(y)
    
    # cls = class k
    for cls in unique_classes:
        count = np.sum(y==cls)
        priors[cls] = count/y_len   
    
    return priors
    
def calculate_feature_probablities(x, y):
    probablity_vectors = {}
    for k in np.unique(y):
    
        '''Calculate feature probablities where class k = 1(spam)'''
        feature_counts = np.zeros(x.shape[1])

        class_count = 0
        
        for row_index, label in enumerate(y):
            if label == k:
                class_count+=1
                current_row = x[row_index]
                for feature_index, value in enumerate(current_row):
                    if value == 1:
                        feature_counts[feature_index]+=1
        probablity_vectors[k] = (feature_counts + 1)/ (class_count + 2)
    
    return probablity_vectors


'''Predict a new x value to be class-0 or class-1'''
def prediction(x_new, priors, feature_probs):
    x_new = np.asarray(x_new)
    scores = []
    classes = list(priors.keys())
    
    for k in classes:
        # calculate probablity of x_new being class k
       
        score = np.log(priors[k])

        for feature, probablity in zip(x_new, feature_probs[k]):
            if feature == 1:
                score+=np.log(probablity)
            else:
                score+=np.log(1 - probablity)
        
        scores.append(score)
    best_index = np.argmax(scores)
    return classes[best_index]
            
prior_dict = calculate_priors(y1)
feature_prob_vectors = calculate_feature_probablities(x1, y1)
                   
p = prediction([1,1,0], prior_dict, feature_prob_vectors)
print(f"{[1,1,0]} belongs to class {p}")



'''OLDER UNIMPROVED WORKING IMPLEMENTATIONS'''

'''Numpy version to calculate final score'''
# score = np.log(prior) + np.sum(
#     np.log(
#         np.where(
#             x_new == 1,
#             feature_probability_vector,
#             1 - feature_probability_vector
#         )
#     )
# )

'''Compare both scores to check which gives better probablity'''
# if score0>score1:
#     print("x_new is class 0 (non-spam)")
# else:
#     print("x_new is class 1 (spam)")

'''Manual Implementation to calculate probablities'''
# feature_probablity_vector = np.zeros(x.shape[1])
# for index, i in enumerate(feature_counts):
#     feature_probablity_vector[index] = i/class_count_1

    
    




        
    