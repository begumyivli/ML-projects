import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, Y, w):
    
    for label in np.unique(Y):
        plt.scatter(X[Y == label, 1], X[Y == label, 2], label=f"Class {label}")
    
    # calculating decision boundary
    x_min = np.min(X[:, 1])
    x_max = np.max(X[:, 1])
    x_values = np.array([x_min, x_max])
    y_values = -(w[0] + w[1] * x_values) / w[2]
    
    plt.plot(x_values, y_values, label="Decision Boundary")
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()