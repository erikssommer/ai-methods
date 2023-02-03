import numpy as np


def normalize(matrix):
    return matrix / np.sum(matrix)


def forward(T, O, prob):
    return normalize(np.dot(np.dot(O, T.T), prob))


def main():

    # Transition matrix
    T = np.array([[0.7, 0.3], [0.3, 0.7]])

    # Observation matrices
    O_true = np.array([[0.9, 0.0], [0.0, 0.2]])

    O_false = np.array([[0.1, 0.0], [0.0, 0.8]])

    # Observations given
    evidence = np.array([True, True, False, True, True])

    # Initial probability
    prob = np.array([0.5, 0.5])

    for i, e in enumerate(evidence):
        prob = forward(T, O_true, prob) if e else forward(T, O_false, prob)
        print(f"Probability after {i + 1} observation(s): {prob}")
        

if __name__ == "__main__":
    main()
