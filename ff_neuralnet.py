import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class FirstFFNetwork:
  def __init__(self):
    # First layer and biases
    self.w111 = np.random.randn()
    self.w112 = np.random.randn()
    self.w121 = np.random.randn()
    self.w122 = np.random.randn()
    self.b11 = 0
    self.b12 = 0
    # Second layer weights and bises
    self.w211 = np.random.randn()
    self.w212 = np.random.randn()
    self.b21 = 0
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  
  def feed_forward(self, x):
    self.x1, self.x2 = x
    # First layer pre (axx) and post-activation (hxx)
    self.a11 = self.w111 * self.x1 + self.w112 * self.x2 + self.b11
    self.h11 = self.sigmoid(self.a11)
    self.a12 = self.w121 * self.x1 + self.w122 * self.x2 + self.b12
    self.h12 = self.sigmoid(self.a12)
    # Second layer pre (axx) and post-activation (hxx)
    self.a21 = self.w211 * self.h11 + self.w212 * self.h12 + self.b21
    self.h21 = self.sigmoid(self.a21)
    return self.h21
  
  def back_propagation(self, x, y):
    self.feed_forward(x)
    # Second layer errors (mse)
    self.dw211 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h11
    self.dw212 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h12
    self.db21 = (self.h21 - y) * self.h21 * (1 - self.h21)
    # First layer errors (mse)
    self.dw111 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.w211 * self.h11 * (1 - self.h11) * self.x1
    self.dw112 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.w211 * self.h11 * (1 - self.h11) * self.x2
    self.db11 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.w211 * self.h11 * (1 - self.h11)
    self.dw121 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.w212 * self.h12 * (1 - self.h12) * self.x1
    self.dw122 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.w212 * self.h12 * (1 - self.h12) * self.x2
    self.db12 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.w212 * self.h12 * (1 - self.h12)
  
  def fit(self, X, Y, epochs=1, learning_rate=1, init=True, display_loss=False):
    if init:
      self.__init__()
    if display_loss:
      losses = {}
    for i in tqdm(range(epochs), total=epochs, unit='epoch'):
      dw111, dw112, dw121, dw122, dw211, dw212, db11, db12, db21 = [0]*9
      for x, y in zip(X, Y):
        self.back_propagation(x, y)
        dw111 += self.dw111
        dw112 += self.dw112
        dw121 += self.dw121
        dw122 += self.dw122
        dw211 += self.dw211
        dw212 += self.dw212
        db11 += self.db11
        db12 += self.db12
        db21 += self.db21
      m = X.shape[1]
      self.w111 -= learning_rate * dw111 / m
      self.w112 -= learning_rate * dw112 / m
      self.w121 -= learning_rate * dw121 / m
      self.w122 -= learning_rate * dw122 / m
      self.w211 -= learning_rate * dw211 / m
      self.w212 -= learning_rate * dw212 / m
      self.b11 -= learning_rate * db11 / m
      self.b12 -= learning_rate * db12 / m
      self.b21 -= learning_rate * db21 / m
      if display_loss:
        Y_pred = self.predict(X)
        losses[i] = mean_squared_error(Y_pred, Y)
    if display_loss:
      plt.plot(np.array(list(losses.values())).astype(float))
      plt.xlabel('Epochs')
      plt.ylabel('Mean Squared Errors')
      plt.show()
  
  def predict(self, X):
    Y_pred = []
    for x in X:
      y_pred = self.feed_forward(x)
      Y_pred.append(y_pred)
    return np.array(Y_pred)
