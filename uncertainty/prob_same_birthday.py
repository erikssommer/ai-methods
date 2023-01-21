import random
import math


def task1():
    # The value of N when the probability will be greater than 50%
    n_values = range(10, 50)
    prob_min_50 = 0
    min_n = None

    for n in n_values:
        prob = same_birthday(n)
        if prob >= 0.5:
            prob_min_50 = prob
            if not min_n:
                min_n = n

    proportion = prob_min_50 / len(n_values)

    # Print the results
    print("The proportion of N where the event happens with the least 50% chance is:", proportion)
    print("The smallest N where the probability of the event occurring is at least 50% is:", min_n)


def same_birthday(n):
    same_birthday = 0
    iterations = 10000

    for _ in range(iterations):
        birthdays = []
        for _ in range(n):
            birthdays.append(random.randint(1, 365))
        if len(birthdays) != len(set(birthdays)):
            same_birthday += 1
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
