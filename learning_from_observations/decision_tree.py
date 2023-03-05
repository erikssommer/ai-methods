import numpy as np
from pathlib import Path
from typing import Tuple
import graphviz as gv
import math

# Run 'pip install graphviz'


class Node:
    """ Node class used to build the decision tree"""

    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)


def visualize_tree(node, graph=None):
    """Visualizes the decision tree using graphviz"""
    if graph is None:
        graph = gv.Digraph()
        graph.attr("node", shape="box")
    if node.attribute is None:
        graph.node(str(id(node)), str(node.value), shape="oval")
    else:
        graph.node(str(id(node)), str(node.attribute))
        for value, child_node in node.children.items():
            child_id = str(id(child_node))
            graph.edge(str(id(node)), child_id, label=str(value))
            visualize_tree(child_node, graph)
    return graph


def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    if measure == "random":
        # Allocate a random number as importance to each attribute
        # Return the attribute with highest importance
        return attributes[np.argmax(np.random.rand(len(attributes)))]
    elif measure == "information_gain":
        # Using the fact that there are only two possible values for each attribute
        # Count the number of examples where the 8th element is 2
        positive_examples = sum(example[7] == 2 for example in examples)

        # Calculate the total entropy of the dataset
        total_entropy = entropy(
            positive_examples, len(examples) - positive_examples)

        # Calculate the information gain for each attribute
        attribute_info_gains = []

        # Iterate over all attributes
        for attribute in attributes:
            # Count the number of examples for each possible value of the attribute
            feature_counts = [
                sum(example[attribute] == i for example in examples) for i in [1, 2]]

            # Calculate the entropy of the dataset for each possible value of the attribute
            feature_entropies = [entropy(sum((example[attribute] == i) and (
                example[7] == 2) for example in examples), count) for count in feature_counts]

            # Calculate the information gain for this attribute
            attribute_info_gain = information_gain(total_entropy, examples, feature_counts, feature_entropies)
            attribute_info_gains.append(attribute_info_gain)

        # Return the attribute with the highest information gain
        return attributes[np.argmax(attribute_info_gains)]

def information_gain(total_entropy, examples, feature_counts, feature_entropies) -> float:
    # Calculate the weighted average of feature entropies
    weighted_entropies = [count / len(examples) * entropy for count, entropy in zip(feature_counts, feature_entropies)]
    weighted_entropy_sum = sum(weighted_entropies)
    
    # Calculate the information gain for this attribute
    attribute_info_gain = total_entropy - weighted_entropy_sum
    
    # Return the information gain
    return attribute_info_gain

def entropy(positive_examples, negative_examples):
    # Calculate the probability of a positive example
    q = positive_examples / (positive_examples + negative_examples)

    # Calculate the entropy of the dataset using the binary entropy formula
    try:
        entropy = -(q * math.log2(q) + (1 - q) * math.log2(1 - q))
    except ValueError:
        # Handle the case where q is zero or one, which would cause a math domain error
        entropy = 0

    # Return the entropy of the dataset
    return entropy


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node's parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # Implemented the steps of the pseudocode in Figure 19.5 on page 678
    # Some changes due to the node creation and linking above
    if len(examples) == 0:
        # If examples is empty, return the plurality value of parent_examples
        node.value = plurality_value(parent_examples)
    elif len(np.unique(examples[:, -1])) == 1:
        # If all examples have the same label, return the label
        node.value = examples[0, -1]
    elif len(attributes) == 0:
        # If attributes is empty, return the plurality value of examples
        node.value = plurality_value(examples)
    else:
        # Choose the attribute with highest importance
        A = importance(attributes, examples, measure)
        # Set the node's attribute to the chosen attribute
        node.attribute = A
        # For each unique value of the chosen attribute

        # Using the fact that there are only two possible values for each attribute
        exs_1 = [example for example in examples if example[A] == 1]
        exs_2 = [example for example in examples if example[A] != 1]

        # Recursively learn the decision tree
        learn_decision_tree(np.array(exs_1), np.delete(
            attributes, np.where(attributes == A)), examples, node, 1, measure)
        learn_decision_tree(np.array(exs_2), np.delete(
            attributes, np.where(attributes == A)), examples, node, 2, measure)

    # The node is returned when the recursion is finished
    return node


def get_attribute_values(attribute, examples):
    """ Returns the unique values of the attribute in the examples """
    return set(examples[:, attribute])


def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test


if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "random"

    for i in range(2):
        # Set the number of trials to run
        num_trials = 100
        # List to store training accuracies
        train_accuracies = []
        # List to store test accuracies
        test_accuracies = []

        for i in range(num_trials):
            tree = learn_decision_tree(examples=train,
                                       attributes=np.arange(
                                           0, train.shape[1] - 1, 1, dtype=int),
                                       parent_examples=None,
                                       parent=None,
                                       branch_value=None,
                                       measure=measure)

            train_accuracy = accuracy(tree, train)
            test_accuracy = accuracy(tree, test)

            # Append the training accuracy to the list
            train_accuracies.append(train_accuracy)
            # Append the test accuracy to the list
            test_accuracies.append(test_accuracy)

            if i == num_trials-1:
                # Visualize the last tree
                graph = visualize_tree(tree)
                graph.render("tree")

        # Calculate the mean of the training accuracies
        mean_train_accuracy = np.mean(train_accuracies)
        # Calculate the mean of the test accuracies
        mean_test_accuracy = np.mean(test_accuracies)
        # Calculate the variance of the training accuracies
        var_train_accuracy = np.var(train_accuracies)
        # Calculate the variance of the test accuracies
        var_test_accuracy = np.var(test_accuracies)

        print(f"Measure: {measure}")
        print(f"Mean Training Accuracy: {mean_train_accuracy}")
        print(f"Mean Test Accuracy: {mean_test_accuracy}")
        print(f"Variance Training Accuracy: {var_train_accuracy}")
        print(f"Variance Test Accuracy: {var_test_accuracy}")

        # Set the measure to the other measure
        if measure == "random":
            measure = "information_gain"
