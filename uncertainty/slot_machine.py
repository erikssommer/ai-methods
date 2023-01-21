import numpy as np
import matplotlib.pyplot as plt


def slot_machine(coins, p_win):
    # Initialize an empty list to store the number of plays until broke
    plays_until_broke = []

    # Run the simulation for a large number of iterations
    for i in range(10000):
        # Start with 10 coins
        current_coins = coins
        plays = 0
        # Play until broke
        while current_coins > 0:
            # Increase the number of plays
            plays += 1
            # Check if we win or lose
            if np.random.rand() < p_win:
                current_coins += 1
            else:
                current_coins -= 1
        # Add the number of plays to the list
        plays_until_broke.append(plays)

    # Calculate the mean and median number of plays
    mean_plays = np.mean(plays_until_broke)
    median_plays = np.median(plays_until_broke)

    return plays_until_broke, mean_plays, median_plays

def plot(play_until_broke, mean_plays, median_plays):
    # Plot a histogram of the number of plays
    plt.hist(plays_until_broke, bins=50)
    plt.title('Plays until broke')
    plt.xlabel('Number of plays')
    plt.ylabel('Rate of occurrence')
    plt.axvline(x=mean_plays, color='r', linestyle='dashed', label='Mean')
    plt.axvline(x=median_plays, color='g', linestyle='dotted', label='Median')
    plt.legend()
    plt.show()

if __name__ == '__main__':
     # Set the initial number of coins and the probability of winning
    coins = 10
    p_win = 11/64
    plays_until_broke, mean_plays, median_plays = slot_machine(coins, p_win)

    # Print the results
    print("Mean number of plays: ", mean_plays)
    print("Median number of plays: ", median_plays)

    # Plot the results
    plot(plays_until_broke, mean_plays, median_plays)