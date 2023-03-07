import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with random values from a normal distribution
        self.weights_input_hidden = np.random.normal(
            0.0, pow(self.input_size, -0.5), (self.hidden_size, self.input_size))
        self.weights_output_hidden = np.random.normal(
            0.0, pow(self.hidden_size, -0.5), (self.output_size, self.hidden_size))

    def sigmoid(self, x):
        # Activation function for the hidden layer
        return 1 / (1 + np.exp(-x))

    def linear(self, x):
        # Activation function for the output layer
        return x

    def forward(self, inputs):
        # Calculate the output of the hidden layer
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)

        # Calculate the output of the output layer
        final_inputs = np.dot(self.weights_output_hidden, hidden_outputs)
        final_outputs = self.linear(final_inputs)

        return final_outputs

    def train(self, inputs, targets):
        # Forward pass
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(self.weights_output_hidden, hidden_outputs)
        final_outputs = self.linear(final_inputs)

        # Calculate the loss
        loss = self.loss(targets, final_outputs)

        # Backward pass
        # Calculate output layer error
        output_errors = targets - final_outputs

        # Calculate the gradient of the weights between the hidden and output layers
        # using the chain rule and the derivative of the linear activation function of the output layer
        gradient_hidden_output = output_errors * hidden_outputs.T

        # Update the weights between the hidden and output layers by moving in the direction of the negative gradient
        self.weights_output_hidden += self.learning_rate * gradient_hidden_output

        # Calculate the hidden layer error
        hidden_errors = np.dot(self.weights_output_hidden.T,
                               output_errors) * hidden_outputs * (1 - hidden_outputs)

        # Calculate the gradient of the weights between the input and hidden layers
        # Using the chain rule and the derivative of the sigmoid activation function of the hidden layer
        gradient_input_hidden = hidden_errors * inputs.T

        # Update the weights between the input and hidden layers by moving in the direction of the negative gradient
        self.weights_input_hidden += self.learning_rate * gradient_input_hidden

        # Return the loss
        return loss

    def predict(self, input):
        # Predict the output of a single input
        input = input.reshape(((self.input_size, self.output_size)))
        return self.forward(input)[0]

    def loss(self, y_true, y_pred):
        # (1/2) * Σ(y_true - y_pred)^2
        return 0.5 * np.sum(np.square(y_true - y_pred))

    def mse(self, y_true, y_pred):
        # Calculate the mean squared error
        return np.mean(((y_true - y_pred) ** 2))


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # Hyperparameters
    input_size = 2
    hidden_size = 2
    output_size = 1
    learning_rate = 0.1
    n_epochs = 1000

    # Create neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Train neural network
    for i in range(n_epochs):
        for j in range(len(X_train)):
            inputs = X_train[j].reshape((input_size, output_size))
            targets = y_train[j]

            loss = nn.train(inputs, targets)

        if i % 100 == 0 and i != 0:
            print("Epoch:", i, "Loss:", loss)

    # Evaluate on training set
    y_train_pred = np.array([nn.predict(X_train[i])
                            for i in range(len(X_train))])
    # Using the mean squared error as the loss function
    mse_train = nn.mse(y_train, y_train_pred)
    print("MSE on training set:", mse_train)

    # Evaluate on test set
    y_test_pred = np.array([nn.predict(X_test[i]) for i in range(len(X_test))])
    # Using the mean squared error as the loss function
    mse_test = nn.mse(y_test, y_test_pred)
    print("MSE on test set:", mse_test)
