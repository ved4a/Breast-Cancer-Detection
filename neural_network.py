import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, lambda_reg=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        # Xavier method weight init
        limit_hidden = np.sqrt(6 / (input_size + hidden_size))
        limit_output = np.sqrt(6 / (hidden_size + output_size))
        self.W1 = np.random.uniform(-limit_hidden, limit_hidden, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.uniform(-limit_output, limit_output, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y, y_hat):
        data_loss = -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
        reg_loss = (self.lambda_reg / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss

    def backprop(self, X, y, y_hat):
        N = y.shape[0]

        delta2 = y_hat - y.reshape(-1, 1)
        self.dW2 = np.dot(self.a1.T, delta2) / N + self.lambda_reg * self.W2
        self.db2 = np.sum(delta2, axis=0) / N

        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        self.dW1 = np.dot(X.T, delta1) / N + self.lambda_reg * self.W1
        self.db1 = np.sum(delta1, axis=0) / N

    def update_weights(self):
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

    def train(self, X_train, y_train, X_val, y_val, max_iters=500, patience=10):
        train_losses, val_losses, accuracies = [], [], []
        best_val_loss = float('inf')
        wait = 0

        for i in range(max_iters):
            y_train_hat = self.forward(X_train)
            train_loss = self.compute_loss(y_train, y_train_hat)
            self.backprop(X_train, y_train, y_train_hat)
            self.update_weights()

            y_val_hat = self.forward(X_val)
            val_loss = self.compute_loss(y_val, y_val_hat)

            val_acc = np.mean((y_val_hat > 0.5).astype(int).flatten() == y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            accuracies.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at iteration {i}")
                    break

            if i % 10 == 0:
                print(f"Iteration {i}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        return train_losses, val_losses, accuracies

    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int)