import numpy as np
import matplotlib.pyplot as plt

""" 
Implement a feedforward neural network with one hidden layer that supports regression. The neural network needs to take two input features. The hidden layer needs to support the sigmoid activation function with two units. Finally, the output layer with one unit. To train the neural network with one hidden layer, implement the gradient descent algorithm with a loss function and its derivative. Implement the mean squared error you get on the training and test set using the trained model. The training input has the format X_train = [[ 0.09762701 0.43037873][0.20552675 0.08976637][-0.1526904 0.29178823]], and the targets have the format y_train = [0.06759077 0.09010412 0.06689603]
"""


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
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate):
        self.learning_rate = learning_rate
        self.input_weight = np.random.normal(size=(input_dim, hidden_dim))
        self.output_weight = np.random.normal(size=(hidden_dim, output_dim))
        self.input_bias = np.random.normal(size=(1, hidden_dim))
        self.output_bias = np.random.normal(size=(1, output_dim))

    def perceptron(self, x):
        return np.dot(x, self.input_weight) + self.input_bias

    def sigmoid(self, x):
        # Activation function for the hidden layer
        return 1 / (1 + np.exp(-x))
    
    def linear(self, x):
        # Activation function for the output layer
        return x

    def loss(self, target, output):
        return np.mean(np.square(target - output))
    
    def loss_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true)

    def forward_pass(self, x):
        hidden_layer = self.sigmoid(self.perceptron(x))
        output_layer = np.dot(hidden_layer, self.output_weight) + self.output_bias
        final_output = self.linear(output_layer)

        return hidden_layer, final_output
        

    def train(self, input, target, epochs):
        for i in range(epochs):
            hidden_layer, output_layer = self.forward_pass(input)

            target = target.reshape((len(target), 1))

            d_output = self.loss_derivative(target, output_layer)

            d_weights_output = np.dot(hidden_layer.T, d_output)
            d_bias_output = np.sum(d_output)

            d_hidden = np.dot(d_output, self.output_weight.T) * hidden_layer * (1 - hidden_layer)
            d_weights_hidden = np.dot(input.T, d_hidden)
            d_bias_hidden = np.sum(d_hidden, axis=0)

            self.input_weight -= self.learning_rate * d_weights_hidden
            self.input_bias -= self.learning_rate * d_bias_hidden
            self.output_weight -= self.learning_rate * d_weights_output
            self.output_bias -= self.learning_rate * d_bias_output

            _, y_pred = self.forward_pass(input)
            if (i+1) % 10000 == 0:
                print("Epoch:", i+1, "Loss:", self.loss(target, y_pred))

        return y_pred
    
    def predict(self, x):
        _, y_pred = self.forward_pass(x)
        return y_pred


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    nn = NeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1, learning_rate=0.0001)

    # Predict before training
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)

    # Print loss before training
    print("Train loss:", nn.loss(y_train, y_train_pred))
    print("Test loss:", nn.loss(y_test, y_test_pred))

    nn.train(X_train, y_train, epochs=100000)

    # Predict after training
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)

    # Print loss after training
    print("Train loss:", nn.loss(y_train, y_train_pred))
    print("Test loss:", nn.loss(y_test, y_test_pred))

    plt.scatter(y_test, y_test_pred)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.show()

    plt.scatter(range(len(y_test)), y_test)
    plt.scatter(range(len(y_test_pred)), y_test_pred, color='red')
    plt.xlabel('Index')
    plt.ylabel('Predictions')
    plt.show()