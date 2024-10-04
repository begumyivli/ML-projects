import numpy as np
import pandas as pd
import math

def fit(X, y):
    parameters = {}
    for label in [0,1]:
        X_label = X[y == label]
        parameters[label] = {
            'mean': X_label.mean(axis=0),
            'var': X_label.var(axis=0) + 1e-4  # to avoid zero division error
        }
    return parameters

def predict(row, parameters):
    labels = [0, 1]
    posteriors = []

    for label in labels:
        prior = len(X_train[y_train == label]) / len(X_train)

        likelihood = 1
        for i in range(len(row)):
            mean = parameters[label]['mean'][i]
            var = parameters[label]['var'][i]
            # row[i] = feature value
            numerator = math.exp(- (row[i] - mean) ** 2 / (2 * var))
            denominator = math.sqrt(2 * math.pi * var)

            likelihood *= numerator / denominator

        posterior = prior * likelihood
        posteriors.append(posterior)
    
    if posteriors[0] > posteriors[1]:
        return 0
    else:
        return 1


data = pd.read_csv('wdbc.data', header=None)
# column 1 is the Diagnosis column
data[1] = data[1].apply(lambda x: 1 if x == 'M' else 0)

# column 0 is the ID column
X = data.drop([0,1], axis=1).values
y = data[1].values


np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split_idx = int(X.shape[0] * 0.7)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# training
parameters = fit(X_train, y_train)

# test accuracy
y_pred_test = []
for row in X_test:
    prediction = predict(row, parameters)
    y_pred_test.append(prediction)
y_pred_test = np.array(y_pred_test)

test_accuracy = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {test_accuracy}")

# train accuracy
y_pred_train = []
for row in X_train:
    prediction = predict(row, parameters)
    y_pred_train.append(prediction)
y_pred_train = np.array(y_pred_train)

train_accuracy = np.mean(y_pred_train == y_train)
print(f"Training Accuracy: {train_accuracy}")