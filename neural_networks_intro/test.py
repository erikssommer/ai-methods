import numpy as np

def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def linear(self, x):
        return x

    def forward(self, X):
        # Compute output for the given input
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.linear(self.z2)
        return self.y_hat
    
    def mse_loss(self, y, y_hat):
        # Calculate Mean Squared Error loss
        return np.mean((y - y_hat)**2)
    
    def mse_loss_deriv(self, y, y_hat):
        # Calculate derivative of Mean Squared Error loss w.r.t y_hat
        return 2 * (y_hat - y) / y.shape[0]

    def backward(self, X, y, y_hat, learning_rate):
        # Backpropagate the error
        delta3 = self.mse_loss_deriv(y, y_hat)
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for i in range(epochs):
            y_hat = self.forward(X)
            loss = self.mse_loss(y, y_hat)
            self.backward(X, y, y_hat, learning_rate)
            if i % 100 == 0:
                print(f"Epoch {i}, loss = {loss}")
    
    def predict(self, X):
        return self.forward(X)
    

if __name__ == "__main__":
    np.random.seed(0)
    # Generate data
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # Create a neural network object
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    # Train the neural network using the training data
    nn.train(X_train, y_train, learning_rate=0.1, epochs=1000)

    # Test the neural network using the test data
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)

    # Calculate Mean Squared Error on training and test sets
    mse_train = nn.mse_loss(y_train, y_pred_train)
    mse_test = nn.mse_loss(y_test, y_pred_test)

    print(f"Mean Squared Error on training set: {mse_train:.4f}")
    print(f"Mean Squared Error on test set: {mse_test:.4f}")


