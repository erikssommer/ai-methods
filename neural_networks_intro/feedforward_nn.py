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

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with random values from a normal distribution
        self.weights_input_hidden = np.random.normal(0.0, pow(self.input_size, -0.5), (self.hidden_size, self.input_size))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.hidden_size, -0.5), (self.output_size, self.hidden_size))

    def forward(self, inputs):
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = sigmoid(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = final_inputs

        return final_outputs

    def train(self, inputs, targets):
        # Forward pass
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = sigmoid(hidden_inputs)

        # Output layer
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = final_inputs

        # Backward pass
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors) * hidden_outputs * (1 - hidden_outputs)

        # Update weights
        self.weights_hidden_output += self.learning_rate * np.dot(output_errors, hidden_outputs.T)
        self.weights_input_hidden += self.learning_rate * np.dot(hidden_errors, inputs.T)

    def predict(self, inputs):
        inputs = inputs.reshape(((self.input_size, self.output_size)))
        return self.forward(inputs)[0]
    
    """
    def calculate_loss(self, inputs, targets):
        # Forward pass
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = sigmoid(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = final_inputs

        # Calculate loss
        loss = np.mean((targets - final_outputs) ** 2)

        return loss
    """


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # Hyperparameters
    input_size = 2
    hidden_size = 2
    output_size = 1
    learning_rate = 0.01
    n_epochs = 1000

    # Create neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Train neural network
    for i in range(n_epochs):
        for j in range(len(X_train)):
            inputs = X_train[j].reshape((2, 1))
            targets = y_train[j]

            nn.train(inputs, targets)


    # Evaluate on training set
    y_train_pred = np.array([nn.predict(X_train[i]) for i in range(len(X_train))])
    mse_train = np.mean((y_train - y_train_pred) ** 2)
    print("MSE on training set:", mse_train)

    # Evaluate on test set
    y_test_pred = np.array([nn.predict(X_test[i]) for i in range(len(X_test))])
    mse_test = np.mean((y_test - y_test_pred) ** 2)
    print("MSE on test set:", mse_test)

    """
    # Calculate loss on training set
    loss_train = nn.calculate_loss(X_train.T, y_train)
    print("Loss on training set:", loss_train)

    # Calculate loss on test set
    loss_test = nn.calculate_loss(X_test.T, y_test)
    print("Loss on test set:", loss_test)
    """

    # Plot predictions
    plt.scatter(y_train, y_train_pred)
    plt.plot([0, 5], [0, 5], '--', color='gray')  # Add diagonal line for reference
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Predicted vs True values (Training set)')
    plt.show()

    # Plot predicted vs true values for test set
    plt.scatter(y_test, y_test_pred)
    plt.plot([0, 5], [0, 5], '--', color='gray')  # Add diagonal line for reference
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('Predicted vs True values (Test set)')
    plt.show()

