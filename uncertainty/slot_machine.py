import numpy as np
import matplotlib.pyplot as plt

# Set the initial number of coins and the probability of winning
coins = 10
p_win = 11/64

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

# Print the results
print("Mean number of plays: ", mean_plays)
print("Median number of plays: ", median_plays)

# Plot a histogram of the number of plays
plt.hist(plays_until_broke, bins=50)
plt.xlabel('Plays')
plt.ylabel('Frequency')
plt.show()