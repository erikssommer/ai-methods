import numpy as np

# Normalize a matrix
def normalize(matrix):
    return matrix / np.sum(matrix)


# Forward algorithm
def forward(T, O, prob):
    return normalize(np.dot(np.dot(O, T), prob))


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

    # List of probabilities
    probs = []
    probs.append(prob)

    # Compute probabilities
    for _, value in enumerate(evidence):
        prob = forward(T, O_true, prob) if value else forward(T, O_false, prob)
        probs.append(prob)

    # Convert to numpy array
    probs = np.array(probs)

    # Print probabilities
    print(probs)


if __name__ == "__main__":
    main()
