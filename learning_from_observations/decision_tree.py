import numpy as np
from pathlib import Path
from typing import Tuple
import graphviz as gv

# Run pip install graphviz


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
        # Allocate a random number as importance to each attribute and return the attribute with highest importance
        return argmax(attributes, lambda _: np.random.random())
    elif measure == "information_gain":
        # Calculate the information gain of each attribute and return the attribute with highest importance
        return argmax(attributes, lambda a: information_gain(examples, a))


def argmax(iterable, key=None):
    return np.argmax([key(x) if key else x for x in iterable])


def entropy(examples):
    """ Calculates entropy of examples """
    labels = examples[:, -1]
    entropy = 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        prob = label_count / labels.size
        entropy += -prob * np.log2(prob)
    return entropy


def information_gain(examples, attribute):
    """ Calculates information gain of attribute on examples """
    gain = entropy(examples)
    for v in get_attribute_values(attribute, examples):
        exs = examples[examples[:, attribute] == v]
        gain -= exs.shape[0] / examples.shape[0] * entropy(exs)
    return gain


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
        for v in get_attribute_values(A, examples):
            # Examples with the value v for the chosen attribute
            exs = examples[examples[:, A] == v]
            # Recursively learn the decision tree
            learn_decision_tree(exs, np.delete(
                attributes, A), examples, node, v, measure)

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

    num_trials = 100  # set the number of trials to run
    train_accuracies = []  # create an empty list to store training accuracies
    test_accuracies = []  # create an empty list to store test accuracies

    for i in range(num_trials):
        tree = learn_decision_tree(examples=train,
                                attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                                parent_examples=None,
                                parent=None,
                                branch_value=None,
                                measure=measure)

        train_accuracy = accuracy(tree, train)
        test_accuracy = accuracy(tree, test)

        train_accuracies.append(train_accuracy)  # append the training accuracy to the list
        test_accuracies.append(test_accuracy)  # append the test accuracy to the list

        mean_train_accuracy = sum(train_accuracies) / num_trials  # calculate the mean of the training accuracies
        mean_test_accuracy = sum(test_accuracies) / num_trials  # calculate the mean of the test accuracies

        if i == num_trials:
            # Visualize the last tree
            graph = visualize_tree(tree)
            graph.render("tree")

    print(f"Mean Training Accuracy: {mean_train_accuracy}")
    print(f"Mean Test Accuracy: {mean_test_accuracy}")
