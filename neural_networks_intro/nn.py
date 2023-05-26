import numpy as np
from tqdm import trange

#######################################################
#                                                     #
#             Functions to generate data              #
#                                                     #
#######################################################


def func(X: np.ndarray) -> np.ndarray:
    """The data generating function"""
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """Add noise to the data generating function"""
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Provide training and test data for training the neural network"""
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test


#######################################################
#                                                     #
#             Feed-forward Neural Network             #
#                                                     #
#######################################################

class Neuron:
    def __init__(self, in_size: int, activation: str) -> None:
        # Initialize weights uniformly between -0.5 and 0.5
        # in_size + 1 since we have bias term
        self.weights = np.random.uniform(-0.5, 0.5, in_size + 1)

        if activation == "sigmoid":
            # The sigmoid activation function
            self.activation = lambda x: 1 / (1 + np.exp(-x))
            # The derivative of the Sigmoid activation function
            self.d_activation = lambda x: x * (1 - x)
        elif activation == "linear":
            self.activation = lambda x: x
            self.d_activation = lambda x: 1
        else:
            raise NotImplementedError(
                f"{activation} is not a valid activation function")

        # Declare containers
        self.__input = None
        self.output = None
        self.delta = None

    def __call__(self, x: np.ndarray) -> np.float64:
        """
        Make a forward pass through a neuron.
        :param x: size = (1, n)
        """
        # Store input
        self.__input = x
        # Bias: self.weights[-1], Weights: self.weights[: -1]
        # Linear combination
        linear = np.dot(x, self.weights[: -1]) + self.weights[-1]
        # Applying activation function and store output
        self.output = self.activation(linear)
        return self.output

    def update(self, lr: float) -> None:
        """Update the weights."""
        self.weights += lr * np.append(self.__input, 1) * self.delta
        self.__empty()

    def __empty(self) -> None:
        """Empty all container variables."""
        self.__input, self.output, self.delta = None, None, None


class Layer:
    def __init__(self, in_size: int, n_units: int, activation: str) -> None:
        """Initialize a layer that takes in_size input with n_units units."""
        self.neurons = [Neuron(in_size, activation) for _ in range(n_units)]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Make a forward pass through the layer."""
        return np.array([neuron(X) for neuron in self.neurons])

    def backward(self, input_) -> None:
        """Run backpropagation through the layer."""
        for curr_neuron in self.neurons:
            # If this is the output layer
            if not isinstance(input_, Layer):
                err = input_ - curr_neuron.output
            # If this is the hidden layer
            else:
                output_unit = input_.neurons[0]
                err = output_unit.weights[self.neurons.index(
                    curr_neuron)] * output_unit.delta

            # Calculate curr_neuron's delta
            curr_neuron.delta = curr_neuron.d_activation(
                curr_neuron.output) * err

    def update(self, lr: float):
        """Update all weights in the layer"""
        for neuron in self.neurons:
            neuron.update(lr)


class NeuralNetwork:
    def __init__(self, in_size: int, n_units: int, lr: float) -> None:
        super(NeuralNetwork, self).__init__()
        self.lr = lr

        # Construct the neural network with given input size, hidden layer size
        # and 1 output unit
        self.layers = [Layer(in_size, n_units, "sigmoid"),
                       Layer(in_size, 1, "linear")]

    def predict(self, x: np.ndarray) -> float:
        """Make a forward pass through the neural network."""
        for layer in self.layers:
            x = layer(x)
        return x.item()

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict(x) for x in X])

    def train(self, X, y, epochs: int) -> None:
        """Train the neural network for `epochs` number of epochs.
        To train the neural network, we use the backpropagation algorithm."""
        for _ in trange(epochs, desc='Epoch'):
            for i in range(len(X)):
                # Propagate the inputs forward to compute the outputs
                self.predict(X[i])
                # Propagate deltas backward from output layer to input layer
                self.backward(y[i])
                # Update every weight in the network using deltas
                self.update()

    def compute_mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean((self.predict_batch(X) - y) ** 2)

    def backward(self, target_y_i) -> None:
        """Run the backward pass through the neural network."""
        input_ = target_y_i
        for layer in reversed(self.layers):
            layer.backward(input_)
            input_ = layer

    def update(self):
        """Update the weights of all layers in the network."""
        for layer in self.layers:
            layer.update(self.lr)


if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    model = NeuralNetwork(X_train.shape[1], 2, lr=1e-1)
    model.train(X_train, y_train, epochs=600)
    print(f"Training MSE: {model.compute_mse(X_train, y_train):.6f}")
    print(f"Test MSE: {model.compute_mse(X_test, y_test):.6f}")
