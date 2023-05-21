import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """The function to approximate"""
    return x ** 2 - np.cos(x)

def f_derivative(x):
    """The derivative of the function to approximate"""
    return 2 * x + np.sin(x)

def newtons_method(x):
    return x - (f(x) / f_derivative(x))


if __name__ == "__main__":
    epochs = 100
    x0 = 0.1

    for _ in range(epochs):
        x0 = newtons_method(x0)
    
    print(x0)

    # Plotting the graph
    x = np.linspace(0, 1, 1000)
    plt.plot(x, f(x), label="f(x)")

    # Plot y = 0 line
    plt.axhline(y=0, color='black', linestyle='--')

    # Plot x0 on the graph
    plt.scatter(x0, f(x0), color='red', marker='o', label="solution")

    plt.legend()
    plt.show()
