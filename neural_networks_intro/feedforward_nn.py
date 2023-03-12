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


class FeedforwardNeuralNetwork:
    """Feedforward neural network with one hidden layer"""

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.learning_rate = learning_rate

        # Initialize weights and biases with random values from a normal distribution
        self.weights_input_hidden = np.random.normal(size=(input_dim, hidden_dim))
        self.weights_hidden_output = np.random.normal(size=(hidden_dim, output_dim))
        self.bias_hidden = np.random.normal(size=(1, hidden_dim))
        self.bias_output = np.random.normal(size=(1, output_dim))

    def sigmoid(self, x):
        """Activation function for the hidden layer"""
        return 1 / (1 + np.exp(-x))

    def linear(self, x):
        """Activation function for the output layer"""
        return x

    def loss(self, target, output):
        """Mean squared error loss function"""
        return np.mean(np.square(target - output))

    def loss_derivative(self, y_true, y_pred):
        """Derivative of the mean squared error loss function"""
        return 2 * (y_pred - y_true)

    def forward(self, x):
        """Forward pass"""
        # Calculate the output of the hidden layer
        hidden_inputs = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        # Calculate the output of the output layer
        final_inputs = np.dot(
            hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_output = self.linear(final_inputs)

        return hidden_outputs, final_output

    def train_step(self, input, target):
        """Train step for learning the weights and biases"""
        hidden_layer, output_layer = self.forward(input)

        # Reshape the target to match the output layer
        target = target.reshape((len(target), 1))

        # Backward pass
        # Calculate output layer error
        # The gradient is the derivative of the loss function
        output_error = self.loss_derivative(target, output_layer)

        # Calculate the gradient of the weights and biases between the hidden and output layers
        # Using the chain rule and the derivative of the linear activation function of the output layer
        gradient_weights_output = np.dot(hidden_layer.T, output_error)
        gradient_bias_output = np.sum(output_error)

        # Calculate hidden layer error
        hidden_error = np.dot(
            output_error, self.weights_hidden_output.T) * hidden_layer * (1 - hidden_layer)

        # Calculate the gradient of the weights and biases between the input and hidden layers
        # Using the chain rule and the derivative of the sigmoid activation function of the hidden layer
        gradient_weights_hidden = np.dot(input.T, hidden_error)
        gradient_bias_hidden = np.sum(hidden_error, axis=0)

        # Update weights and biases using gradient descent moving in the direction of the gradient
        self.weights_input_hidden -= self.learning_rate * gradient_weights_hidden
        self.bias_hidden -= self.learning_rate * gradient_bias_hidden
        self.weights_hidden_output -= self.learning_rate * gradient_weights_output
        self.bias_output -= self.learning_rate * gradient_bias_output

        # Return the target to match the output layer for the loss function
        return target

    def fit(self, input, target, epochs):
        """Train the neural network"""
        for i in range(epochs):
            target = self.train_step(input, target)
            _, y_pred = self.forward(input)

            # Print the loss every 10% of the epochs
            if (i+1) % (epochs // 10) == 0:
                print(f"Epoch: {i+1}, Loss: {self.loss(target, y_pred)}")

    def predict(self, input):
        """Predict the output for the given input"""
        _, y_pred = self.forward(input)
        return y_pred


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # Hyperparameters
    input_dim = 2
    hidden_dim = 2
    output_dim = 1
    learning_rate = 0.0001
    n_epochs = 100000

    fnn = FeedforwardNeuralNetwork(input_dim, hidden_dim, output_dim, learning_rate)

    # Predict before training
    y_train_pred = fnn.predict(X_train)
    y_test_pred = fnn.predict(X_test)

    # Print mse before training
    print("Mean squared error before training:")
    print("MSE on training set:", fnn.loss(y_train, y_train_pred))
    print("MSE on test set:", fnn.loss(y_test, y_test_pred))

    # Train the neural network
    print("\nTraining...")
    fnn.fit(X_train, y_train, n_epochs)

    # Predict after training
    y_train_pred = fnn.predict(X_train)
    y_test_pred = fnn.predict(X_test)

    # Print mse after training
    print("\nMean squared error after training:")
    print("MSE on training set:", fnn.loss(y_train, y_train_pred))
    print("MSE on test set:", fnn.loss(y_test, y_test_pred))