import numpy as np
import idx2numpy
import time
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from skimage import exposure

#************************************DATA PREPARATION*************************************
# read
file_train_images = 'train-images.idx3-ubyte'
file_train_labels = 'train-labels.idx1-ubyte'
file_test_images = 't10k-images.idx3-ubyte'
file_test_labels = 't10k-labels.idx1-ubyte'

X_train_all = idx2numpy.convert_from_file(file_train_images)
y_train_all = idx2numpy.convert_from_file(file_train_labels)
X_test_all = idx2numpy.convert_from_file(file_test_images)
y_test_all = idx2numpy.convert_from_file(file_test_labels)


# f: flattenedi n: normalized
# Filter for digits 2, 3, 8, 9
digits = [2, 3, 8, 9]
train_mask = np.isin(y_train_all, digits)
test_mask = np.isin(y_test_all, digits)

X_train_raw = X_train_all[train_mask]
y_train_raw = y_train_all[train_mask]
X_test_raw = X_test_all[test_mask]
y_test_raw = y_test_all[test_mask]

# Flatten the images
X_train_f = X_train_raw.reshape(X_train_raw.shape[0], -1)
X_test_f = X_test_raw.reshape(X_test_raw.shape[0], -1)

# normalize
def normalize_data(X):
    X_normalized = X.astype(float)
    X_normalized /= 255.0
    return X_normalized

# Normalize the training and test data
X_train_fn = normalize_data(X_train_f)
X_test_fn = normalize_data(X_test_f)

X_train = X_train_fn
X_test = X_test_fn
y_train = y_train_raw
y_test = y_test_raw


#****************************************2***************************************
#Remove comment to apply Feature Extraction
"""
def extract_hog_features(images):
    hog_features = []
    for image in images:
        fd, hog_image = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train_raw) // if you split sample data like [::5] in data preparation part,
X_test_hog = extract_hog_features(X_test_raw)   // then you should also split these 2 lines

X_train = X_train_hog
X_test = X_test_hog
"""

#****************************************1.a***************************************
print()
print("RESULTS FOR 1.A")

def svm_train(X, y, digit, C):
    n_samples, n_features = X.shape
    y_transformed = np.where(y == digit, 1, -1)  # making the chosen digit's label 1 and others -1

    d = n_features

    Q = np.zeros((d + 1, d + 1))
    Q[1:, 1:] = np.eye(n_features) * C

    p = np.zeros(d + 1)
    c = -np.ones(n_samples)

    Q = matrix(Q,)
    p = matrix(p)
    c = matrix(c)

    # stack a column of ones to X for the bias term in the constraint matrix
    X_aug = np.hstack((np.ones((n_samples, 1)), X))
    A = np.vstack([y_transformed[i] * X_aug[i] for i in range(n_samples)])
    A = matrix(-A)

    # QP problem solving
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 40
    solution = solvers.qp(Q, p, A, c)

    # weights and bias
    weights = np.array(solution['x']).flatten()
    b = weights[0]
    w = weights[1:]

    return w, b

def svm_predict(w, b, X):
    return np.sign(np.dot(X, w) + b)


start_time = time.time()

models = []
classes_to_keep = [2, 3, 8, 9]
for digit in classes_to_keep:
    w, b = svm_train(X_train, y_train, digit, 1.1)
    models.append((digit, w, b))
    
end_time = time.time()
time_taken = end_time - start_time
print(f"Training Time: {time_taken:.4f} seconds")


train_accuracies = []
test_accuracies = []

for digit, weights, bias in models:
    # Training accuracy
    y_train_pred = svm_predict(weights, bias, X_train)
    train_correct = (y_train_pred == 1) & (y_train == digit)
    train_incorrect = (y_train_pred == 1) & (y_train != digit)
    train_accuracy = np.sum(train_correct) / (np.sum(train_correct) + np.sum(train_incorrect))
    train_accuracies.append((digit, train_accuracy))
    
    # Test accuracy
    y_test_pred = svm_predict(weights, bias, X_test)
    test_correct = (y_test_pred == 1) & (y_test == digit)
    test_incorrect = (y_test_pred == 1) & (y_test != digit)
    test_accuracy = np.sum(test_correct) / (np.sum(test_correct) + np.sum(test_incorrect))
    test_accuracies.append((digit, test_accuracy))

print("Training accuracies for each digit:")
for digit, acc in train_accuracies:
    print(f"Digit {digit}: {acc:.2f}")

print("Test accuracies for each digit:")
for digit, acc in test_accuracies:
    print(f"Digit {digit}: {acc:.2f}")
print()
#****************************************1.b***************************************
print()
print("RESULTS FOR 1.B")

# Deciding C with 5 fold cv
def tune_hyperparameters(X_train, y_train):
    svm_gscv = GridSearchCV(LinearSVC(dual = False), {'C': [0.01, 0.1, 1, 10, 100]}, cv=3)
    svm_gscv.fit(X_train, y_train)
    best_C = svm_gscv.best_params_['C']
    return best_C

svm_classifier = LinearSVC(dual = False, C=0.01)
start_time = time.time()
svm_classifier.fit(X_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
print(f"1.b Training Time: {time_taken:.4f} seconds")

train_predictions = svm_classifier.predict(X_train)
test_predictions = svm_classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

#****************************************1.c***************************************
print()
print("RESULTS FOR 1.C")

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def kernel_matrix(X1, X2, gamma):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = rbf_kernel(X1[i], X2[j], gamma)
    return K
    
def svm_train_dual(X, y, digit, C=5, gamma=0.6):
    n_samples, n_features = X.shape
    y_transformed = np.where(y == digit, 1, -1)
    
    Q = matrix(np.outer(y_transformed, y_transformed) * kernel_matrix(X, X, gamma))
    p = matrix(np.ones(n_samples) * -1)
    G = matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    A = matrix(y_transformed.astype(float).reshape(1, -1)) 
    b = matrix([0.0])  

    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 40
    solution = solvers.qp(Q, p, G, h, A, b)
    
    alphas = np.array(solution['x']).flatten()
    sv = alphas > 1e-5
    alphas_sv = alphas[sv]
    sv_y = y_transformed[sv]
    sv_X = X[sv]

    weighted_sv = alphas_sv[:, np.newaxis] * sv_y[:, np.newaxis] * sv_X
    w = np.sum(weighted_sv, axis=0)
    b = np.mean([y_k - np.dot(w, x_k) for (y_k, x_k) in zip(sv_y, sv_X)])

    return w, b


def svm_predict(w, b, X):
    y_predict = np.dot(X, w) + b
    return np.sign(y_predict)

models = []
classes_to_keep = [2, 3, 8, 9]
for digit in classes_to_keep:
    w, b = svm_train_dual(X_train, y_train, digit, C=5, gamma=0.6)
    models.append((digit, w, b))

train_accuracies = []
test_accuracies = []

for digit, w, b in models:
    # Training accuracy
    y_train_pred = svm_predict(w, b, X_train)
    train_accuracy = np.mean(y_train_pred == np.where(y_train == digit, 1, -1))
    train_accuracies.append((digit, train_accuracy))

    # Test accuracy
    y_test_pred = svm_predict(w, b, X_test)
    test_accuracy = np.mean(y_test_pred == np.where(y_test == digit, 1, -1))
    test_accuracies.append((digit, test_accuracy))

print("Training accuracies for each digit:")
for digit, acc in train_accuracies:
    print(f"Digit {digit}: {acc:.2f}")

print("Test accuracies for each digit:")
for digit, acc in test_accuracies:
    print(f"Digit {digit}: {acc:.2f}")

#****************************************1.d***************************************
print()
print("RESULTS FOR 1.D")

def tune_hyperparameters_dual(X_train, y_train):
    svm_dual_gscv = GridSearchCV(SVC(kernel='rbf'), {'C': [0.1, 1, 10, 100]}, cv=3, scoring='accuracy')
    svm_dual_gscv.fit(X_train, y_train)
    best_C = svm_dual_gscv.best_params_['C']
    return best_C

svm_classifier_dual = SVC(kernel='rbf', C= 100)
start_time = time.time()
svm_classifier_dual.fit(X_train, y_train)
end_time = time.time()
time_taken = end_time - start_time
print(f"1.d Training Time: {time_taken:.4f} seconds")

train_predictions_rbf = svm_classifier_dual.predict(X_train)
test_predictions_rbf = svm_classifier_dual.predict(X_test)

train_accuracy_rbf = accuracy_score(y_train, train_predictions_rbf)
test_accuracy_rbf = accuracy_score(y_test, test_predictions_rbf)

print(f'Training Accuracy: {train_accuracy_rbf}')
print(f'Test Accuracy: {test_accuracy_rbf}')

#****************************************3***************************************

def visualize_support_vectors(svm_model, start_idx=1000, end_idx=1020):
    support_vectors = svm_model.support_vectors_
    if end_idx is None:
        end_idx = len(support_vectors) 
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten() 
    for i, ax in enumerate(axes):
        idx = start_idx + i
        if idx < end_idx:
            ax.imshow(support_vectors[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.show()
    
# visualize_support_vectors(svm_classifier_dual)