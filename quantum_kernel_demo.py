# Chaotic Quantum Kernel Demo (FIXED)

# Author: Aimé T Shangula

# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------

# Chaotic Feature Map

# -----------------------------

def chaotic_feature_map(X, beta=0.7, freq=3.0):
  """
  Stable chaotic feature embedding inspired by quantum feature maps.
  Avoids numerical instability and undefined zeta regions.

  ```
  X : array [n_samples, n_features]
  beta : chaos strength
  freq : Fourier frequency multiplier
  """
  X = np.asarray(X)

  # Fourier-style interference term
  fourier = np.sin(freq * np.pi * X) + np.cos(freq * np.pi * X**2)

  # Analytic chaotic surrogate (zeta-inspired but numerically stable)
  # Uses logarithmic and power-law coupling instead of raw zeta
  chaos = np.sum(np.log1p(np.abs(X)) ** beta, axis=1, keepdims=True)

  return np.hstack([fourier, chaos])


# -----------------------------

# Dataset

# -----------------------------

X, y = make_moons(n_samples=400, noise=0.12, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42
)

# -----------------------------

# Apply Feature Map

# -----------------------------

X_train_mapped = chaotic_feature_map(X_train)
X_test_mapped = chaotic_feature_map(X_test)

# -----------------------------

# Train SVM (Linear kernel on chaotic space)

# -----------------------------

clf = SVC(kernel="linear", C=1.0)
clf.fit(X_train_mapped, y_train)

y_pred = clf.predict(X_test_mapped)
print(classification_report(y_test, y_pred))

# -----------------------------

# Visualization

# -----------------------------

def plot_decision_boundary(model, X, y):
  h = 0.02
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


  xx, yy = np.meshgrid(
      np.arange(x_min, x_max, h),
      np.arange(y_min, y_max, h)
  )

  grid = np.c_[xx.ravel(), yy.ravel()]
  grid_scaled = scaler.transform(grid)
  Z = model.predict(chaotic_feature_map(grid_scaled))
  Z = Z.reshape(xx.shape)

  plt.figure(figsize=(6, 5))
  plt.contourf(xx, yy, Z, alpha=0.75)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=30)
  plt.title("Chaotic Quantum-Inspired Kernel SVM")
  plt.tight_layout()
  plt.show()


plot_decision_boundary(clf, X_test, y_test)
