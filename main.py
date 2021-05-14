import csv
import math
import copy
import random


class Node:
    def __init__(self, at_id=None, child=None, leaf_cl=None, data_s=None, at_key=None, at_as=None):
        self.attribute_id = at_id
        self.children = child
        self.leaf_class = leaf_cl
        self.data_set = data_s
        self.attributes_set = at_key
        self.attribute_associated = at_as

    def decision_tree(self):
        return id3(self, self.attributes_set, self.data_set)


def read_file(file_name):
    with open(file_name) as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        data_records = []

        for record in data_reader:
            for i in range(len(record)):
                element_to_check_if_digit = record[i]
                if element_to_check_if_digit.replace('.', '', 1).isdigit():
                    record[i] = float(record[i])

            data_records.append(record)

        data_records.pop(0)
        return data_records


def get_attributes_set(data_records, attributes):
    filtered_attributes_set = attributes
    for i in range(1, len(attributes)):
        unique_attributes_list = []
        for i1 in range(len(data_records)):
            is_attribute_unique = True
            for x in unique_attributes_list:
                if x == data_records[i1][i]:
                    is_attribute_unique = False
                    break

            if is_attribute_unique:
                unique_attributes_list.append(data_records[i1][i])

        if len(unique_attributes_list) > (len(data_records) / 2):
            filtered_attributes_set[i] = 0

        if len(unique_attributes_list) > 10 and filtered_attributes_set[i] != 0:
            unique_attributes_list.sort()
            discrete_id_base = math.ceil(len(unique_attributes_list) / 10)
            discrete_values_of_attribute = []
            for i1 in range(1, 10):
                discrete_id = discrete_id_base * i1
                discrete_values_of_attribute.append(unique_attributes_list[discrete_id])
            discrete_values_of_attribute.append(unique_attributes_list[len(unique_attributes_list) - 1])
            filtered_attributes_set[i] = discrete_values_of_attribute

    return filtered_attributes_set


def discrete_with_attributes_set(data_records, attributes):
    for i in range(len(attributes)):
        if isinstance(attributes[i], list):
            for x in data_records:
                for i1 in range(len(attributes[i])):
                    if not x[i] > attributes[i][i1]:
                        x[i] = i1
                        break
    return data_records


def calculate_entropy(set_to_calculate):
    occurrences_of_classes = []
    for i in range(len(set_to_calculate)):
        is_class_unique = True
        for x in occurrences_of_classes:
            if x[0] == set_to_calculate[i][0]:
                x[1] += 1
                is_class_unique = False
                break

        if is_class_unique:
            occurrences_of_classes.append([set_to_calculate[i][0], 1])

    entropy = 0
    for i in range(len(occurrences_of_classes)):
        entropy += -((occurrences_of_classes[i][1] / len(set_to_calculate)) * math.log(occurrences_of_classes[i][1]
                                                                                       / len(set_to_calculate), 2))

    return entropy


def calculate_info_gain(attribute_number, set_to_calculate):
    occurrences_of_attribute = []
    for i in range(len(set_to_calculate)):
        is_attribute_unique = True
        for x in occurrences_of_attribute:
            if x[0] == set_to_calculate[i][attribute_number]:
                x[1] += 1
                x[2].append(set_to_calculate[i])
                is_attribute_unique = False
                break

        if is_attribute_unique:
            occurrences_of_attribute.append([set_to_calculate[i][attribute_number], 1, [set_to_calculate[i]]])

    occurrences_of_attribute.sort(key=lambda k: k[0])
    inf = 0
    for i in range(len(occurrences_of_attribute)):
        entropy_for_attribute = calculate_entropy(occurrences_of_attribute[i][2])
        inf += (len(occurrences_of_attribute[i][2]) / len(set_to_calculate)) * entropy_for_attribute

    info_gain = calculate_entropy(set_to_calculate) - inf

    return info_gain


def id3(node, attributes, learning_set):
    is_learning_set_same_class = True
    classes_list = []

    for i in range(len(learning_set)):
        classes_list.append(learning_set[i][0])
        if learning_set[i][0] != learning_set[0][0]:
            is_learning_set_same_class = False

    if is_learning_set_same_class:
        node.leaf_class = classes_list[0]
        return node

    are_all_attributes_checked = True
    for i in range(len(attributes)):
        if attributes[i] != 0:
            are_all_attributes_checked = False
            break

    if are_all_attributes_checked:
        most_probable_class = max(set(classes_list), key=classes_list.count)
        node.leaf_class = most_probable_class
        return node

    current_info_gain = -99999
    attribute_with_best_info_gain = None

    for i in range(len(attributes)):
        if attributes[i] != 0:
            info_gain_for_attribute = calculate_info_gain(i, learning_set)
            if info_gain_for_attribute > current_info_gain:
                attribute_with_best_info_gain = i
                current_info_gain = info_gain_for_attribute

    split_learning_set = split_by_attribute(learning_set, attributes, attribute_with_best_info_gain)
    node.attribute_id = attribute_with_best_info_gain
    node.children = []

    for i in range(len(split_learning_set)):
        child_node = Node(at_key=split_learning_set[i][1], data_s=split_learning_set[i][0],
                          at_as=split_learning_set[i][2])
        node.children.append(child_node)

    for i in range(len(node.children)):
        node.children[i].decision_tree()

    return node


def split_by_attribute(set_to_split, attributes_key, attribute_number):
    new_attributes_key = copy.deepcopy(attributes_key)
    new_attributes_key[attribute_number] = 0
    split_sets = []
    for i in range(len(set_to_split)):
        is_attribute_unique = True
        for x in split_sets:
            if x[2] == set_to_split[i][attribute_number]:
                x[0].append(set_to_split[i])
                is_attribute_unique = False
                break

        if is_attribute_unique:
            split_sets.append([[set_to_split[i]], new_attributes_key, set_to_split[i][attribute_number]])

    return split_sets


def evaluate_with_decision_tree(tree, data_set):
    predicted_correct = 0
    predicted_wrong = 0
    for i in range(len(data_set)):
        prediction = get_leaf_value(tree, data_set[i])
        if data_set[i][0] == prediction:
            predicted_correct += 1
        else:
            predicted_wrong += 1

    accuracy_of_predictions = (predicted_correct / (predicted_correct + predicted_wrong)) * 100
    return accuracy_of_predictions


def get_leaf_value(tree, data_record):
    if tree.leaf_class is None:
        for i in range(len(tree.children)):
            if data_record[tree.attribute_id] == tree.children[i].attribute_associated:
                return get_leaf_value(tree.children[i], data_record)
    else:
        return tree.leaf_class


def create_decision_tree(data_set, attributes_key):
    validate_set_len = math.ceil(len(data_set) / 5)
    best_tree = []
    current_tree_accuracy = 0
    for i in range(5):
        validate_set = []
        learn_set = []
        for i1 in range(len(data_set)):
            if (i * validate_set_len) <= i1 < ((i + 1) * validate_set_len):
                validate_set.append(data_set[i1])
            else:
                learn_set.append(data_set[i1])

        tree = Node(at_key=attributes_key, data_s=learn_set)
        tree.decision_tree()
        tree_accuracy = evaluate_with_decision_tree(tree, validate_set)
        if tree_accuracy > current_tree_accuracy:
            best_tree = tree
            current_tree_accuracy = tree_accuracy

    return best_tree


titanic_data_records = read_file('titanic.csv')
attributes_set = []

for n in range(len(titanic_data_records[0])):
    attributes_set.append(1)
attributes_set[0] = 0

attributes_set = get_attributes_set(titanic_data_records, attributes_set)
discrete_titanic_data_records = discrete_with_attributes_set(titanic_data_records, attributes_set)

tree_accuracy_list = []
for n in range(100):
    random.shuffle(discrete_titanic_data_records)
    test_set = []
    learning_and_validation_set = []

    for n1 in range(len(discrete_titanic_data_records)):
        if n1 < (len(discrete_titanic_data_records) * 0.2):
            test_set.append(discrete_titanic_data_records[n1])
        else:
            learning_and_validation_set.append(discrete_titanic_data_records[n1])

    decision_tree = create_decision_tree(learning_and_validation_set, attributes_set)
    decision_tree_accuracy = evaluate_with_decision_tree(decision_tree, test_set)
    tree_accuracy_list.append(round(decision_tree_accuracy, 2))

print("Decision tree average accuracy: ", round(sum(tree_accuracy_list) / len(tree_accuracy_list),2), "%")
print("Decision tree best accuracy: ", round(max(tree_accuracy_list), 2), "%")
print("Decision tree worst accuracy: ", round(min(tree_accuracy_list), 2), "%")
