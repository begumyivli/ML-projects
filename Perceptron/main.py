import numpy as np

from perceptron import perceptron_learning_algorithm
from plot import plot_decision_boundary

data = np.load('data_small.npy')
label = np.load('label_small.npy')

w_learned = perceptron_learning_algorithm(data, label)

print("Iteration Num: ", w_learned[1])
plot_decision_boundary(data,label,w_learned[0])
