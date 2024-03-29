import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """The function to approximate"""
    return 2 * x**4 - 5 * x**3 + 2 * x**2 - 6 * x + 4


def f_derivative(x):
    """The derivative of the function to approximate"""
    return 8 * x**3 - 15 * x**2 + 4 * x - 6


def gradient_descent(lr, epochs):
    """Gradient descent algorithm"""
    # Start at x=0
    x = 0

    # Iteratively apply the gradient descent update rule
    for _ in range(epochs):
        # Compute the derivative
        derivative = f_derivative(x)

        # Update x
        x -= lr * derivative

    return x


if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.0025
    epochs = 1000

    res = gradient_descent(learning_rate, epochs)

    print("The local minimum occurs at {:.4f}".format(res))

    # Plot f(x) and the derivative
    x = np.linspace(0, 3, 1000)
    plt.plot(x, f(x), label="f(x)")
    plt.plot(x, f_derivative(x), label="f'(x)")
    plt.scatter(res, f(res), color='red', marker='o', label="Minimum")
    plt.title("Gradient Descent")
    # Plot y = 0 line
    plt.axhline(y=0, color='black', linestyle='--')
    # Plot res as a vertical line
    plt.axvline(x=res, color='black', linestyle='--')
    plt.legend()
    plt.show()

