import random

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


def main():
    # the value of N when the probability will be greater than 50%
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

if __name__ == "__main__":
    main()
