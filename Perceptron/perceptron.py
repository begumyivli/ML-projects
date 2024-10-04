import numpy as np

def perceptron_learning_algorithm(data, labels, max_iter=10000):
    n_samples, n_features = data.shape

    # weights initializing to zeros
    w = np.zeros(n_features)
    iteration_num = 0
    for iter in range(max_iter):
        misclassified_point_exists = False
        for i in range(n_samples): # each sample
            if np.sign(labels[i]) != np.sign(np.dot(w, data[i])):
                misclassified_point_exists = True
                #w(t+1) --> w(t) + (x_*, y_*)
                w += labels[i] * data[i]
        if not misclassified_point_exists:
            break # all points are correctly classified
        iteration_num += 1
    
    return w, iteration_num
