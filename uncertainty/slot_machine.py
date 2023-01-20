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

"""
def play_slot_machine(coins):
    outcome = random.choices(["BAR", "BELL", "LEMON", "CHERRY"], k=3)
    if outcome == ["BAR", "BAR", "BAR"]:
        return 20
    elif outcome == ["BELL", "BELL", "BELL"]:
        return 15
    elif outcome == ["LEMON", "LEMON", "LEMON"]:
        return 5
    elif outcome == ["CHERRY", "CHERRY", "CHERRY"]:
        return 3
    elif outcome[:2] == ["CHERRY", "CHERRY"]:
        return 2
    elif "CHERRY" in outcome:
        return 1
    else:
        return -1

def simulate(coins, n_sims, max_plays):
    plays_until_broke = []
    for i in range(n_sims):
        coins_copy = coins
        plays = 0
        while coins_copy > 0 and plays < max_plays:
            coins_won = play_slot_machine(coins_copy)
            coins_copy += coins_won
            plays += 1
        if coins_copy <= 0:
            plays_until_broke.append(plays)
    return plays_until_broke

plays_until_broke = simulate(10, 10000, 10000)
mean_plays = sum(plays_until_broke) / len(plays_until_broke)
median_plays = sorted(plays_until_broke)[len(plays_until_broke) // 2]
print("Mean number of plays until broke:", mean_plays)
print("Median number of plays until broke:", median_plays)

# Plotting histogram
plt.hist(plays_until_broke, bins=range(min(plays_until_broke), max(plays_until_broke) + 2, 1))
plt.xlabel('Plays until broke')
plt.ylabel('Frequency')
plt.title('Simulation of plays until broke')
plt.show()
"""