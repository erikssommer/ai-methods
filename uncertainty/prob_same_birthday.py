import random
import math


def task1():
    # The value of N when the probability will be greater than 50%
    n_values = range(10, 50)
    prob_min_50 = 0
    min_n = None

    # Iterate over the values of n
    for n in n_values:
        # Calculate the probability of the event occurring
        prob = same_birthday(n)
        # Check if the probability is greater than 50%
        if prob >= 0.5:
            prob_min_50 = prob
            # Check if the value of n is smaller than the current value then update the value
            if not min_n:
                min_n = n

    # Calculate the proportion of N where the event happens with the least 50% chance
    proportion = prob_min_50 / len(n_values)

    # Print the results
    print("The proportion of N where the event happens with the least 50% chance is:", proportion)
    print("The smallest N where the probability of the event occurring is at least 50% is:", min_n)


def same_birthday(n):
    """ 
    Function that takes N and computes the probability of the event via simulation
    """
    # The number of times the event occurs
    same_birthday = 0
    iterations = 10000

    # Simulate over the number of iterations
    for _ in range(iterations):
        birthdays = []
        # Generate a random birthday for each person
        for _ in range(n):
            birthdays.append(random.randint(1, 365))
        # Check if there are any duplicate birthdays
        if len(birthdays) != len(set(birthdays)):
            same_birthday += 1
    # Return the probability of the event occurring
    return same_birthday / iterations


def task2():
    # The number of trials to run
    trials = 1000
    total_people = 0
    for _ in range(trials):
        total_people += make_group()

    # The average number of people to add to the group
    average_people = total_people / trials

    # Rounding up to the nearest integer
    print("The expected number of people to add to the group:",
          math.ceil(average_people))


def make_group():
    group = []
    days = set(range(1, 365))
    counter = 0
    # Iterate until all days are used
    while len(days) > 0:
        # Pick a random day as a birthday
        birthday = random.randint(1, 365)
        # Add the birthday to the group and increment the counter
        group.append(birthday)
        counter += 1
        # Remove the birthday from the set of days
        days.discard(birthday)

    return counter


if __name__ == "__main__":
    print("------ Part 1 -------")
    task1()
    print("------ Part 2 -------")
    task2()
