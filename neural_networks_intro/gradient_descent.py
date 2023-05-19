

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
