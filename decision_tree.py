import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

data = pd.read_csv('wdbc.data', header=None)
# column 1 is the Diagnosis column
data[1] = data[1].apply(lambda x: 1 if x == 'M' else 0)

# column 0 is the ID column
X = data.drop([0, 1], axis=1)
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

decision_tree = DecisionTreeClassifier(criterion="gini", random_state=42)

param_grid = {'max_depth': range(1, 10)}
grid_search = GridSearchCV(decision_tree, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best max depth:", grid_search.best_params_)

best_tree = grid_search.best_estimator_

plt.figure(figsize=(16,8))
plot_tree(best_tree, feature_names=data.columns.drop([0, 1]), class_names=['Benign', 'Malignant'], filled=True)
plt.title('Decision Tree for Breast Cancer Classification')
plt.show()

from sklearn.metrics import accuracy_score
test_predictions = best_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy of Decision Tree:", test_accuracy)

importances = best_tree.feature_importances_
print(importances)
print()
indices = np.argsort(importances)[::-1]
# print(indices)
# print(X.columns)

# Most significant features
from sklearn.linear_model import LogisticRegression
feature_number = [5, 10, 15, 20]
cv_scores = {}

for num_features in feature_number:
    top_features = X.columns[indices[:num_features]]
    
    lr = LogisticRegression(max_iter=1000, random_state=42) # default scoring = accuracy
    lr.fit(X_train[top_features], y_train)
    score = lr.score(X_test[top_features], y_test)
    print(f'Average accuracy with top {num_features} features: {score}')


# Random forest
# number of trees in random forest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

tree_numbers = [i for i in range(10,300,5)]

train_accuracies = []
test_accuracies = []

# test and training performances with the varying number of trees in the forest
for tree_num in tree_numbers:
    rf = RandomForestClassifier(n_estimators=tree_num, random_state=42)
    rf.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, rf.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf.predict(X_test))
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

print(sum(test_accuracies) / len(test_accuracies))
plt.figure(figsize=(10, 6))
plt.plot(tree_numbers, train_accuracies, label='Train Accuracy')
plt.plot(tree_numbers, test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Trees in Forest')
plt.ylabel('Accuracy Scores')
plt.title('The Change in Test and Training Performances vs Varying Number of Trees in the Forest')
plt.legend()
plt.grid(True)
plt.show()


