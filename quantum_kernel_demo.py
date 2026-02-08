import numpy as np
import matplotlib.pyplot as plt  
from sklearn import svm  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import make_moons

# Quantum-inspired chaotic feature map

def chaotic_feature_map(X):
    # Example of a chaotic mapping function using logistic map
    chaos = np.zeros_like(X)
    for i in range(len(X)):
        chaos[i] = 4 * X[i] * (1 - X[i])  # Logistic map
    return chaos

# SVM classifier function

def svm_classifier(X_train, y_train, X_test):
    clf = svm.SVC(kernel='rbf')  # Using RBF kernel
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

# Visualization function

def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Example usage
if __name__ == '__main__':
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    X_transformed = chaotic_feature_map(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
    model = svm_classifier(X_train, y_train, X_test)
    plot_decision_boundary(X_transformed, y, model)  
