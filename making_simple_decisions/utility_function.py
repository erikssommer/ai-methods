import numpy as np
import matplotlib.pyplot as plt


def U(x, R):
    return -np.exp(-x/R)

# R is a positive constant representing an individual’s risk tolerance
R = 255  # choose a value for the constant R
x = np.linspace(0, 100, num=100)  # create an array of x values
y = U(x, R)  # calculate the corresponding y values

plt.plot(x, y)  # plot the x and y values
plt.xlabel("x")
plt.ylabel("U(x)")
plt.title("Plot of U(x) = -e^(-x/R)")
plt.show()
