import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
import time

# Reading data
file_train_images = 'archive/train-images.idx3-ubyte'
file_train_labels = 'archive/train-labels.idx1-ubyte'
file_test_images = 'archive/t10k-images.idx3-ubyte'
file_test_labels = 'archive/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(file_train_images)
train_labels = idx2numpy.convert_from_file(file_train_labels)
test_images = idx2numpy.convert_from_file(file_test_images)
test_labels = idx2numpy.convert_from_file(file_test_labels)

# Filtering data
classes = [2,3,8,9]

def filter_images(images, labels, classes):
    # Finding indices of labels
    mask = np.isin(labels, classes)
    filtered_images = images[mask]
    filtered_labels = labels[mask]
    return filtered_images, filtered_labels

flt_train_images, flt_train_labels = filter_images(train_images, train_labels, classes)
flt_test_images, flt_test_labels = filter_images(test_images, test_labels, classes)

# Normalizing data
def normalize_data(data):
    normalized_data = data.astype('float32') / 255.0  # Scale pixel values to [0, 1]
    return normalized_data

flat_images = flt_train_images.reshape(flt_train_images.shape[0], flt_train_images.shape[1] * flt_train_images.shape[2]) # original data
norm_images = normalize_data(flat_images) # normalized data

flat_test_images = flt_test_images.reshape(flt_test_images.shape[0], flt_test_images.shape[1] * flt_test_images.shape[2]) # original test data
norm_test_images = normalize_data(flat_test_images) # normalized test data

#### K-Means with Euclidean Distance ####

# Initializing centroids randomly
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    return centroids

# Assigning data points to clusters
def assign_clusters(data, centroids):
    distances = cdist(data, centroids, metric='euclidean')
    clusters = np.argmin(distances, axis=1)
    return clusters

# Updating centroids based on cluster means
def update_centroids(data, clusters, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def calculate_sse(data, centroids, clusters):
    sse = 0
    for i in range(len(data)):
        sse += np.linalg.norm(data[i] - centroids[clusters[i]])**2
    return sse

def kmeans(data, k, max_iter=100):
    centroids = initialize_centroids(data, k)
    for i in range(max_iter):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    sse = calculate_sse(data, centroids, clusters)
    return centroids, clusters, sse

#### K-Means with Cosine Similarity ####

def assign_clusters_cos(data, centroids):
    distances = cdist(data, centroids, metric='cosine')
    clusters = np.argmin(distances, axis=1)
    return clusters

def kmeans_cos(data, k, max_iter=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iter):
        clusters = assign_clusters_cos(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    sse = calculate_sse(data, centroids, clusters)
    return centroids, clusters, sse

#### Evaluating Performance ####
 
def evaluate_clustering_accuracy(clusters, true_labels, k):
    label_mapping = {}
    for i in range(k):
        cluster_labels = true_labels[clusters == i]
        most_common_label = np.bincount(cluster_labels).argmax()
        label_mapping[i] = most_common_label
    predicted_labels = np.array([label_mapping[cluster] for cluster in clusters])
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy

def average_metrics(num_runs, k, data, test_data, assign_clusters_func, kmeans_func):
    accuracies = []
    train_accuracies = []
    sse_values = []
    train_times = []

    for _ in range(num_runs):
        start_time = time.time()
        centroids, train_clusters, train_sse = kmeans_func(data, k)
        end_time = time.time()
        time_taken = end_time - start_time

        test_clusters = assign_clusters_func(test_data, centroids)

        train_accuracy = evaluate_clustering_accuracy(train_clusters, flt_train_labels, k)
        test_accuracy = evaluate_clustering_accuracy(test_clusters, flt_test_labels, k)

        accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)
        sse_values.append(train_sse)
        train_times.append(time_taken)

    avg_accuracy = np.mean(accuracies)
    avg_train_accuracy = np.mean(train_accuracies)
    avg_sse = np.mean(sse_values)
    avg_time = np.mean(train_times)

    return avg_accuracy, avg_train_accuracy, avg_sse, avg_time

k = 4
num_runs = 10

# Euclidean distance + original data
# avg_accuracy, avg_train_accuracy, avg_sse, avg_time = average_metrics(num_runs, k, flat_images, flat_test_images, assign_clusters, kmeans)

# print("Average Clustering Test Accuracy (Euclidean - Original):", avg_accuracy)
# print("Average Clustering Train Accuracy (Euclidean - Original):", avg_train_accuracy)
# print("Average SSE (Euclidean - Original):", avg_sse)
# print("Average Training Time (Euclidean - Original):", avg_time, "seconds\n")

# Euclidean distance + normalized data
# avg_accuracy_norm, avg_train_accuracy_norm, avg_sse_norm, avg_time_norm = average_metrics(num_runs, k, norm_images, norm_test_images, assign_clusters, kmeans)

# print("Average Clustering Test Accuracy (Euclidean - Normalized):", avg_accuracy_norm)
# print("Average Clustering Train Accuracy (Euclidean - Normalized):", avg_train_accuracy_norm)
# print("Average SSE (Euclidean - Normalized):", avg_sse_norm)
# print("Average Training Time (Euclidean - Normalized):", avg_time_norm, "seconds\n")

# Cosine similarity + original data
# avg_accuracy_cos, avg_train_accuracy_cos, avg_sse_cos, avg_time_cos = average_metrics(num_runs, k, flat_images, flat_test_images, assign_clusters_cos, kmeans_cos)

# print("Average Clustering Test Accuracy (Cosine - Original):", avg_accuracy_cos)
# print("Average Clustering Train Accuracy (Cosine - Original):", avg_train_accuracy_cos)
# print("Average SSE (Cosine - Original):", avg_sse_cos)
# print("Average Training Time (Cosine - Original):", avg_time_cos, "seconds\n")

# Cosine similarity + normalized data
# avg_accuracy_cos_norm, avg_train_accuracy_cos_norm, avg_sse_cos_norm, avg_time_cos_norm = average_metrics(num_runs, k, norm_images, norm_test_images, assign_clusters_cos, kmeans_cos)

# print("Average Clustering Test Accuracy (Cosine - Normalized):", avg_accuracy_cos_norm)
# print("Average Clustering Train Accuracy (Cosine - Normalized):", avg_train_accuracy_cos_norm)
# print("Average SSE (Cosine - Normalized):", avg_sse_cos_norm)
# print("Average Training Time (Cosine - Normalized):", avg_time_cos_norm, "seconds\n")

#### Feature Extraction ####

def calculate_pca(data):
    cov_matrix = np.cov(data, rowvar=False)
    
    # Performing eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    return eigenvalues, eigenvectors

def determine_num_components(eigenvalues, variance_threshold=0.95):
    # Calculating cumulative explained variance ratio
    explained_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1
    
    return num_components

def perform_pca(data, num_components):
    eigenvalues, eigenvectors = calculate_pca(data)
    
    # Selecting the top eigenvalues/eigenvectors
    projection_matrix = eigenvectors[:, :num_components]
    reduced_data = np.dot(data, projection_matrix)
    
    return reduced_data

eigenvalues, eigenvectors = calculate_pca(norm_images)
num_components = determine_num_components(eigenvalues, variance_threshold=0.95)

reduced_train_data = perform_pca(flat_images, num_components)
reduced_test_data = perform_pca(flat_test_images, num_components)

reduced_train_data_norm = perform_pca(norm_images, num_components)
reduced_test_data_norm = perform_pca(norm_test_images, num_components)

# PCA Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(reduced_train_data_norm[:, 0], reduced_train_data_norm[:, 1], c=flt_train_labels, cmap='viridis', s=5)
# plt.colorbar(label='Digit Label', ticks=range(10))
# plt.title('PCA Visualization of MNIST Image Data')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

#### Evaluating Performance ####

# Euclidean distance + original data
avg_accuracy, avg_train_accuracy, avg_sse, avg_time = average_metrics(num_runs, k, reduced_train_data, reduced_test_data, assign_clusters, kmeans)

print("Average Clustering Test Accuracy (Euclidean - Original):", avg_accuracy)
print("Average Clustering Train Accuracy (Euclidean - Original):", avg_train_accuracy)
print("Average SSE (Euclidean - Original):", avg_sse)
print("Average Training Time (Euclidean - Original):", avg_time, "seconds\n")

# Euclidean distance + normalized data
avg_accuracy_norm, avg_train_accuracy_norm, avg_sse_norm, avg_time_norm = average_metrics(num_runs, k, reduced_train_data_norm,reduced_test_data_norm, assign_clusters, kmeans)

print("Average Clustering Test Accuracy (Euclidean - Normalized):", avg_accuracy_norm)
print("Average Clustering Train Accuracy (Euclidean - Normalized):", avg_train_accuracy_norm)
print("Average SSE (Euclidean - Normalized):", avg_sse_norm)
print("Average Training Time (Euclidean - Normalized):", avg_time_norm, "seconds\n")

# Cosine similarity + original data
avg_accuracy_cos, avg_train_accuracy_cos, avg_sse_cos, avg_time_cos = average_metrics(num_runs, k, reduced_train_data,reduced_test_data, assign_clusters_cos, kmeans_cos)

print("Average Clustering Test Accuracy (Cosine - Original):", avg_accuracy_cos)
print("Average Clustering Train Accuracy (Cosine - Original):", avg_train_accuracy_cos)
print("Average SSE (Cosine - Original):", avg_sse_cos)
print("Average Training Time (Cosine - Original):", avg_time_cos, "seconds\n")

# Cosine similarity + normalized data
avg_accuracy_cos_norm, avg_train_accuracy_cos_norm, avg_sse_cos_norm, avg_time_cos_norm = average_metrics(num_runs, k, reduced_train_data_norm, reduced_test_data_norm, assign_clusters_cos, kmeans_cos)

print("Average Clustering Test Accuracy (Cosine - Normalized):", avg_accuracy_cos_norm)
print("Average Clustering Train Accuracy (Cosine - Normalized):", avg_train_accuracy_cos_norm)
print("Average SSE (Cosine - Normalized):", avg_sse_cos_norm)
print("Average Training Time (Cosine - Normalized):", avg_time_cos_norm, "seconds\n")
