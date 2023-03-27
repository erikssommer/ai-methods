import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


rewards: tuple[float, ...] = (-0.1, -0.1, -0.1, -0.1,
                              -0.1, -1.0, -0.1, -1.0,
                              -0.1, -0.1, -0.1, -1.0,
                              -1.0, -0.1, -0.1, 1.0)

transition_matrix: tuple[tuple[tuple[tuple[float, int]]]] = \
    ((((.9, 0), (.1, 4)), ((.1, 0), (.8, 4), (.1, 1)),
      ((.1, 4), (.8, 1), (.1, 0)), ((.1, 1), (.9, 0))),
     (((.1, 1), (.8, 0), (.1, 5)), ((.1, 0), (.8, 5), (.1, 2)),
      ((.1, 5), (.8, 2), (.1, 1)), ((.1, 2), (.8, 1), (.1, 0))),
     (((.1, 2), (.8, 1), (.1, 6)), ((.1, 1), (.8, 6), (.1, 3)),
      ((.1, 6), (.8, 3), (.1, 2)), ((.1, 3), (.8, 2), (.1, 1))),
     (((.1, 3), (.8, 2), (.1, 7)), ((.1, 2), (.8, 7), (.1, 3)),
      ((.1, 7), (.9, 3)), ((.9, 3), (.1, 2))),
     (((.1, 0), (.8, 4), (.1, 8)), ((.1, 4), (.8, 8), (.1, 5)),
      ((.1, 8), (.8, 5), (.1, 0)), ((.1, 5), (.8, 0), (.1, 4))),
     (((1.0, 5),), ((1.0, 5),), ((1.0, 5),), ((1.0, 5),)),
     (((.1, 2), (.8, 5), (.1, 10)), ((.1, 5), (.8, 10), (.1, 7)),
      ((.1, 10), (.8, 7), (.1, 2)), ((.1, 7), (.8, 2), (.1, 5))),
     (((1.0, 7),), ((1.0, 7),), ((1.0, 7),), ((1.0, 7),)),
     (((.1, 4), (.8, 8), (.1, 12)), ((.1, 8), (.8, 12), (.1, 9)),
      ((.1, 12), (.8, 9), (.1, 4)), ((.1, 9), (.8, 4), (.1, 8))),
     (((.1, 5), (.8, 8), (.1, 13)), ((.1, 8), (.8, 13), (.1, 10)),
      ((.1, 13), (.8, 10), (.1, 5)), ((.1, 10), (.8, 5), (.1, 8))),
     (((.1, 6), (.8, 9), (.1, 14)), ((.1, 9), (.8, 14), (.1, 11)),
      ((.1, 14), (.8, 11), (.1, 6)), ((.1, 11), (.8, 6), (.1, 9))),
     (((1.0, 11),), ((1.0, 11),), ((1.0, 11),), ((1.0, 11),)),
     (((1.0, 12),), ((1.0, 12),), ((1.0, 12),), ((1.0, 12),)),
     (((.1, 9), (.8, 12), (.1, 13)), ((.1, 12), (.8, 13), (.1, 14)),
      ((.1, 13), (.8, 14), (.1, 9)), ((.1, 14), (.8, 9), (.1, 12))),
     (((.1, 10), (.8, 13), (.1, 14)), ((.1, 13), (.8, 14), (.1, 15)),
      ((.1, 14), (.8, 15), (.1, 10)), ((.1, 15), (.8, 10), (.1, 13))),
     (((1.0, 15),), ((1.0, 15),), ((1.0, 15),), ((1.0, 15),)))


def valid_state(state: int) -> bool:
    return isinstance(state, (int, np.signedinteger)) and 0 <= state < 16


def valid_action(action: int) -> bool:
    return isinstance(action, (int, np.signedinteger)) and 0 <= action < 4

# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------- Nothing you need to do or use above this line. -----------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


# Use these constants when you implement the value iteration algorithm.
# Do not change these values, except DETERMINISTIC when debugging.
N_STATES: int = 16
N_ACTIONS: int = 4
EPSILON: float = 1e-8
GAMMA: float = 0.9
DETERMINISTIC: bool = False


def get_next_states(state: int, action: int) -> list[int]:
    """
    Fetches the possible next states given the state and action pair.
    :param state: a number between 0 - 15.
    :param action: an integer between 0 - 3.
    :return: A list of possible next states. Each next state is a number between 0 - 15.
    """
    assert valid_state(state), \
        f"State {state} must be an integer between 0 - 15."
    assert valid_action(action), \
        f"Action {action} must be an integer between 0 - 3."
    next_state_probs = {next_state: trans_prob for trans_prob,
                        next_state in transition_matrix[state][action]}
    if DETERMINISTIC:
        return [max(next_state_probs, key=next_state_probs.get)]
    return next_state_probs.keys()


def get_trans_prob(state: int, action: int, next_state: int) -> float:
    """
    Fetches the transition probability for the next state
    given the state and action pair.
    :param state: an integer between 0 - 15.
    :param action: an integer between 0 - 3.
    :param outcome_state: an integer between 0 - 15.
    :return: the transition probability.
    """
    assert valid_state(state), \
        f"State {state} must be an integer between 0 - 15."
    assert valid_action(action), \
        f"Action {action} must be an integer between 0 - 3."
    assert valid_state(next_state), \
        f"Next state {next_state} must be an integer between 0 - 15."
    next_state_probs = {next_state: trans_prob for trans_prob,
                        next_state in transition_matrix[state][action]}
    # If the provided next_state is invalid.
    if next_state not in next_state_probs.keys():
        return 0.
    if DETERMINISTIC:
        return float(next_state == max(next_state_probs, key=next_state_probs.get))
    return next_state_probs[next_state]


def get_reward(state: int) -> float:
    """
    Fetches the reward given the state. This reward function depends only on the current state.
    In general, the reward function can also depend on the action and the next state.
    :param state: an integer between 0 - 15.
    :return: the reward.
    """
    assert valid_state(state), \
        f"State {state} must be an integer between 0 - 15."
    return rewards[state]


def get_action_as_str(action: int) -> str:
    """
    Fetches the string representation of an action.
    :param action: an integer between 0 - 3.
    :return: the action as a string.
    """
    assert valid_action(action), \
        f"Action {action} must be an integer between 0 - 3."
    return ("left", "down", "right", "up")[action]


def value_iteration() -> list[float]:
    # Initialize the utilities of all states to 0
    U = np.zeros(N_STATES)

    # Run value iteration until convergence
    while True:
        # Initialize the change in utility for this iteration
        delta = 0
        # Update the utility of each state
        for state in range(N_STATES):
            # Calculate the new utility of the state
            new_util = u_value(state, U)
            # Update the change in utility if the new utility is better
            delta = max(delta, abs(new_util - U[state]))
            # Update the utility of this state
            U[state] = new_util

        # Check if the change in utility is small enough to stop iterating
        if delta < (EPSILON * (1 - GAMMA)) / GAMMA:
            break

    # Scale and round the utilities to the desired format
    U = np.round(U / 10, decimals=4)
    # Convert the utilities to a list and return them
    return U.tolist()


def u_value(state: int, U: list[float]) -> float:
    # Find the maximum q-value over all actions from this state
    return max(q_value(state, action, U) for action in range(N_ACTIONS))


def q_value(state: int, action: int, U: list[float]) -> float:
    """ Calculate the maximum expected utility of a given state-action pair """
    # Get the next possible states given the current state and action
    next_states: list[int] = get_next_states(state, action)
    # Initialize q-value to 0
    q = 0
    # Calculate the expected utility of each next state, taking into account the transition probabilities and rewards
    for next_state in next_states:
        trans_prob = get_trans_prob(state, action, next_state)
        expected_utility = get_reward(next_state) + GAMMA * U[next_state]
        q += trans_prob * expected_utility

    # Return the weighted sum of the expected utilities, which is the q-value
    return q


def greedy_policy(state: int, U: list[float]) -> int:
    # Create a list to store the Q-values for each action in the current state
    q_values = []
    # Loop through all possible actions in the current state and compute their Q-values
    for action in range(N_ACTIONS):
        q_values.append(q_value(state, action, U))

    # Find the index of the maximum Q-value, which corresponds to the best action
    best_action = np.argmax(q_values)
    # Return the index of the best action
    return best_action


def display_utilities(utils: list[float]):
    # Reshape the utilities into a 4x4 grid
    grid = np.array(utils).reshape(4, 4)

    # Create a heatmap of the utilities using Seaborn
    sns.heatmap(grid, annot=True, fmt=".4f", cmap="YlGnBu")

    # Add a title to the plot
    plt.title("Utilities")

    # Show the plot
    plt.show()


def display_greedy_policy(U: list[float]):
    # Start at the initial state
    curr_state = 0
    # Create a 4x4 grid to hold the optimal actions
    policy = np.zeros((4, 4))

    # Continue until we reach the goal state
    while True:
        # Store the current state in the policy grid
        x, y = divmod(curr_state, 4)
        # Determine the best action to take in the current state
        policy[x][y] = 1
        # Determine the best action to take in the current state
        next_action = greedy_policy(curr_state, U)
        # Get the next possible states that we could transition to
        next_possible_states = list(get_next_states(curr_state, next_action))
        # Choose the next state with the highest utility value
        curr_state = next_possible_states[np.argmax(
            [U[state] for state in next_possible_states])]

        # If we have reached the goal state, store it in the policy grid and break out of the loop
        if curr_state == 15:
            policy[3][3] = 1
            break

    # Create a heatmap of the policy grid using Seaborn
    sns.heatmap(policy, annot=True, fmt=".0f", cmap="YlGnBu")

    # Add a title to the plot
    plt.title("Greedy Policy")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    utilities = value_iteration()
    display_utilities(utilities)
    display_greedy_policy(utilities)
