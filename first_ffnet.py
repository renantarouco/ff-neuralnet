import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ff_neuralnet import FirstFFNetwork

color_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
original_labels = labels
labels = np.mod(labels, 2)
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0, train_size=0.9)

ffn = FirstFFNetwork()
ffn.fit(X_train, Y_train, epochs=1000, learning_rate=0.01, display_loss=True)

Y_pred_train = ffn.predict(X_train)
Y_pred_train_threshold = (Y_pred_train >= 0.5).astype(int).ravel()

Y_pred_val = ffn.predict(X_val)
Y_pred_val_threshold = (Y_pred_val >= 0.5).astype(int).ravel()

accuracy_train = accuracy_score(Y_pred_train_threshold, Y_train)
accuracy_val = accuracy_score(Y_pred_val_threshold, Y_val)
print('Training accuracy:', accuracy_train)
print('Validation accuracy:', accuracy_val)

plt.scatter(X_val[:,0], X_val[:,1], c=Y_pred_val_threshold, cmap=color_map, s=15*(np.abs(Y_pred_val_threshold - Y_val)+.2))
plt.show()