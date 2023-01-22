import matplotlib.pyplot as plt
import random

def play_slot_machine():
    # Play the slot machine and return the payout
    outcome = random.choices(['BAR', 'BELL', 'LEMON', 'CHERRY'], k=3)
    if outcome == ['BAR', 'BAR', 'BAR']:
        return 20
    elif outcome == ['BELL', 'BELL', 'BELL']:
        return 15
    elif outcome == ['LEMON', 'LEMON', 'LEMON']:
        return 5
    elif outcome == ['CHERRY', 'CHERRY', 'CHERRY']:
        return 3
    elif outcome[0] == 'CHERRY' and outcome[1] == 'CHERRY':
        return 2
    elif outcome[0] == 'CHERRY':
        return 1
    else:
        return -1

def simulate_plays(starting_coins, iterations, max_plays):
    # Simulate n iterations of playing the slot machine
    results = []
    for _ in range(iterations):
        coins = starting_coins
        plays = 0
        while coins > 0 and plays < max_plays:
            coins += play_slot_machine()
            plays += 1
        results.append(plays)
    return results

if __name__ == '__main__':
    # Run the simulation with 10 coins, 1000 iterations, and a max of 100 plays
    simulation_results = simulate_plays(10, 1000, 100)
    
    # Compute the mean and median number of plays
    mean_plays = sum(simulation_results) / len(simulation_results)
    median_plays = sorted(simulation_results)[len(simulation_results) // 2]

    # Print the results
    print(f'Mean number of plays: {mean_plays}')
    print(f'Median number of plays: {median_plays}')

    # Plot a histogram of the simulation results
    plt.hist(simulation_results, bins=range(0, max(simulation_results)+1))
    plt.xlabel('Number of plays')
    plt.ylabel('Frequency')
    plt.title('Distribution of plays until going broke')

    # Add vertical lines for the mean and median
    plt.axvline(mean_plays, color='r', linestyle='--', label='Mean')
    plt.axvline(median_plays, color='g', linestyle='-.', label='Median')
    plt.legend()
    plt.show()