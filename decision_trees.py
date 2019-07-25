import numpy as np

np.random.seed(42)

chi_table = {0.01: 6.635,
             0.005: 7.879,
             0.001: 10.828,
             0.0005: 12.116,
             0.0001: 15.140,
             0.00001: 19.511}


def calc_gini(data):
    """
    Calculate gini impurity measure of a data set.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the data set.
    """
    gini = 0.0
    class_values = np.unique(data[:, -1])
    for i in range(len(class_values)):
        gini += np.power(np.divide(np.where(data[:, -1] == class_values[i])[0].shape[0], data.shape[0]), 2)

    return np.subtract(1, gini)


def calc_entropy(data):
    """
    Calculate the entropy of a data set.

    Input:
    - data: any data set where the last column holds the labels.

    Returns the entropy of the data set.
    """

    entropy = 0.0
    class_values = np.unique(data[:, -1])
    for i in range(len(class_values)):
        weight_in_class = np.divide(np.where(data[:, -1] == class_values[i])[0].shape[0], data.shape[0])
        entropy += np.multiply(weight_in_class, np.log2(weight_in_class))

    return -entropy


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.

    def __init__(self, feature, value, data, num_of_instances, majority):
        self.feature = feature  # column index of criteria being tested
        self.value = value  # value necessary to get a true result
        self.children = []
        self.chi_square = self.chi_square_test(data)
        self.num_of_instances = num_of_instances
        self.majority = majority

    def add_child(self, node):
        self.children.append(node)

    def chi_square_test(self, data):
        probability_zero = np.divide(data[np.where(data[:, -1] == 0.0)].shape[0], data.shape[0])
        probability_one = np.divide(data[np.where(data[:, -1] == 1.0)].shape[0], data.shape[0])
        d_0 = data[np.where(data[:, self.feature] <= self.value)].shape[0]
        d_1 = data[np.where(data[:, self.feature] > self.value)].shape[0]
        p_0 = data[np.where(data[:, self.feature] <= self.value)]
        p_0 = p_0[np.where(p_0[:, -1] == 1.0)].shape[0]
        p_1 = data[np.where(data[:, self.feature] > self.value)]
        p_1 = p_1[np.where(p_1[:, -1] == 1.0)].shape[0]
        n_0 = data[np.where(data[:, self.feature] <= self.value)]
        n_0 = n_0[np.where(n_0[:, -1] == 0.0)].shape[0]
        n_1 = data[np.where(data[:, self.feature] > self.value)]
        n_1 = n_1[np.where(n_1[:, -1] == 0.0)].shape[0]
        chi_square = np.add(
            np.divide(
                np.subtract(p_0, np.multiply(d_0, probability_zero)),
                np.multiply(d_0, probability_zero)),
            np.divide(
                np.subtract(n_0, np.multiply(d_0, probability_one)),
                np.multiply(d_0, probability_one)))
        chi_square += np.add(
            np.divide(
                np.subtract(p_1, np.multiply(d_1, probability_zero)),
                np.multiply(d_1, probability_zero)),
            np.divide(
                np.subtract(n_1, np.multiply(d_1, probability_one)),
                np.multiply(d_1, probability_one)))

        return chi_square


def build_tree(data, impurity, chi_value):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training data set.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    if len(np.unique(data[:, -1])) == 1:
        return np.unique(data[:, -1])[0], data.shape[0]
    else:
        best_feature = 0
        best_threshold = 0
        best_gain = 0
        for i in range(np.subtract(data.shape[1], 1)):
            for j in range(np.subtract(data.shape[0], 1)):
                threshold = np.mean(np.array(data[j][i], data[j + 1][i]))
                child_a = data[np.where(data[:, i] <= threshold)]
                weight_a = np.divide(child_a.shape[0], data.shape[0])
                child_b = data[np.where(data[:, i] > threshold)]
                weight_b = np.divide(child_b.shape[0], data.shape[0])
                gain = impurity(data) \
                    - (np.multiply(weight_a, impurity(child_a))
                        + np.multiply(weight_b, impurity(child_b)))
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_feature = i
        majority = 0.0 if np.divide(data[np.where(data[:, -1] == 0.0)].shape[0], data.shape[0]) > 0.5 else 1.0
        root = DecisionNode(best_feature, best_threshold, data, data.shape[0], majority)
        if chi_value == 1 or root.chi_square < chi_table[chi_value]:
            child_a = build_tree(data[np.where(data[:, best_feature] <= best_threshold)], impurity, chi_value)
            child_b = build_tree(data[np.where(data[:, best_feature] > best_threshold)], impurity, chi_value)
            root.add_child(child_a)
            root.add_child(child_b)
        else:
            class_values = np.unique(data[:, -1])
            best_class = 0
            best_count = 0
            for i in range(len(class_values)):
                temp = np.where(data[:, -1] == class_values[i])[0]
                if temp.shape[0] > best_count:
                    best_count = temp.shape[0]
                    best_class = class_values[i]

            return best_class, data.shape[0]

    return root


def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the data set. Note that the last element
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    if isinstance(node, DecisionNode):
        if instance[node.feature] <= node.value:
            if not isinstance(node.children[0], DecisionNode):
                return node.children[0]
            else:
                pred = predict(node.children[0], instance)
        else:
            if not isinstance(node.children[1], DecisionNode):
                return node.children[1]
            else:
                pred = predict(node.children[1], instance)
    else:
        return node

    return pred


def calc_accuracy(node, data):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given data set.

    Input:
    - node: a node in the decision tree.
    - data: the data set on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given data set (%).
    """
    right_prediction = 0
    wrong_prediction = 0

    for i in range(data.shape[0]):
        prediction = predict(node, data[i])
        if prediction[0] == data[i][-1]:
            right_prediction += 1
        else:
            wrong_prediction += 1

    accuracy = np.multiply(np.divide(right_prediction, np.add(right_prediction, wrong_prediction)), 100)
    return accuracy


def print_tree(node, count):
    """
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    """
    if isinstance(node, DecisionNode):
        print('[X' + str(node.feature) + ' <= ' + str(node.value) + ']')
        print('  ' * count, end='')
        print_tree(node.children[0], count + 1)
        print('  ' * count, end='')
        print_tree(node.children[1], count + 1)
    else:
        print('leaf: [{' + str(node[0]) + ': ' + str(node[1]) + '}]')


def count_nodes(tree):
    queue = [tree]
    count = 1
    while len(queue) > 0:
        if isinstance(queue[0], DecisionNode):
            count += 2
            queue.append(queue[0].children[0])
            queue.append(queue[0].children[1])
            del queue[0]
        else:
            del queue[0]

    return count
