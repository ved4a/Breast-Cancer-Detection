import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    # compute sigmoid
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # use gradient descent to train LR model
    def fit(self, X, y):
        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias

            y_predicted = self.sigmoid(linear_model)

            dw = (1 / samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_probabilities(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def prediction(self, X):
        y_predicted_probabilities = self.predict_probabilities(X)
        return [1 if i > 0.5 else 0 for i in y_predicted_probabilities]