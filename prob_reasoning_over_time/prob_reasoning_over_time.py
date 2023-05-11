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

# -------------- v2 ----------------

# Defined in Section 14.3.1 on Page 492
transition_model = np.array([[.7, .3],
                             [.3, .7]])

# Row 1: when evidence = False
# Row 2: when evidence = True
sensor_model = np.array([[.1, .8],
                         [.9, .2]])


def filtering(evidence) -> np.ndarray:
    forward_messages = []
    # Define the prior distribution
    recursion = np.array([[.5], [.5]])
    for i in range(len(evidence)):
        # Computing Equation 14.5 on Page 485
        recursion = sensor_model * (transition_model @ recursion)
        # Normalizing Equation 14.5 (alpha)
        recursion = recursion / np.sum(recursion, axis=1, keepdims=True)
        # Conditioning on evidence
        # Equation 14.5 condition on evidence before computing
        recursion = recursion[evidence[i]]
        # Saving the forward message
        forward_messages.append(recursion)

    return np.array(forward_messages)


def main_v2() -> None:

    # Debug run: Probability distribution of rain at day two after
    # observing umbrella two days
    evidence_debug = np.array([1,1])
    forward_messages = filtering(evidence_debug)

    day2 = forward_messages[-1]

    print("---- DOING DEBUG RUN FOR FILTERING -----")
    print("Probability distribution for Rain given two observations "
          "of umbrella is [{:.3f}; {:.3f}]\n".format(day2[0], day2[1]))


    # Probability distribution of rain at day 5
    # observing umbrella on day 1, 2, no umbrella on day 3, and umbrella on day 4, 5
    evidence = np.array([1, 1, 0, 1, 1])
    forward_messages = filtering(evidence)
    day5 = forward_messages[-1]
    print( "---- DOING FILTERING OF FIVE OBSERVATIONS -----")
    print("Probability distribution for Rain given observations "
          "is [{:.3f}; {:.3f}]".format(day5[0], day5[1]))

    # Print all forward messages
    print("Forward messages:")
    time = 1
    for forward_message in forward_messages:
        print("Time {}: Forward message = [{:.3f}; {:.3f}]".format(time, forward_message[0],
                                                                   forward_message[1]))
        time+=1

if __name__ == "__main__":
    main() # main_v2()
