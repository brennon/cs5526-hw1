import csv, math, json, random

from pprint import pprint
from libpgm.graphskeleton import GraphSkeleton
from libpgm.nodedata import NodeData
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner
from libpgm.tablecpdfactorization import TableCPDFactorization

def create_nodes_from_header(filename):
    """
        Returns a list of node labels for an NBC from the CSV file at *filename*.
        Assumes that the first row of this file contains unquoted node labels.
    """
    csv_file = open(filename, 'r')
    reader = csv.reader(csv_file)
    nodes = reader.next()
    csv_file.close()
    return nodes

def create_edges_from_nodes(nodes):
    """
        Returns a list of edges for an NBC given the list of nodes in *nodes*.
        Assumes that the first node in *nodes* is the class, and all others are
        features.
    """
    class_label = nodes[0]
    return [[class_label, feature] for feature in nodes[1:]]

def create_observations_from_csv(filename, fieldnames):
    """
        Returns a dictionary of observations from data in the CSV file at *filename*.
        Assumes that the first row of this file contains node labels.
    """
    observations = []
    csv_file = open(filename)
    reader = csv.DictReader(csv_file, fieldnames=fieldnames)
    reader.next()
    for row in reader:
        observation = dict()
        for node in nodes:
            observation[node] = row[node]
        observations.append(observation)
    return observations

def create_graph_skeleton(nodes, edges):
    graphSkeleton = GraphSkeleton()
    graphSkeleton.V = nodes
    graphSkeleton.E = edges
    graphSkeleton.toporder()
    return graphSkeleton

def pristine_bn(V, E, Vdata):
    fresh_bn = DiscreteBayesianNetwork()
    fresh_bn.V = list(V)
    fresh_bn.E = list(E)
    fresh_bn.Vdata = Vdata.copy()
    return fresh_bn

def pristine_fn(V, E, Vdata):
    pristine = pristine_bn(V, E, Vdata)
    return TableCPDFactorization(pristine)

def manual_mutual_information(node_a, node_b, observations, Vdata):
    node_a_values = Vdata[node_a]['vals']
    node_b_values = Vdata[node_b]['vals']

    running_sum = 0

    for a in node_a_values:
        marginal_a_dictionary = {}
        marginal_a_dictionary[node_a] = a
        marginal_a = count_matching_observations(marginal_a_dictionary, observations) / float(len(observations))

        for b in node_b_values:

            joint_query_dictionary = {}
            joint_query_dictionary[node_a] = a
            joint_query_dictionary[node_b] = b
            matching_joint_observations = count_matching_observations(joint_query_dictionary, observations)
            joint = count_matching_observations(joint_query_dictionary, observations) / float(len(observations))

            marginal_b_dictionary = {}
            marginal_b_dictionary[node_b] = b
            marginal_b = count_matching_observations(marginal_b_dictionary, observations) / float(len(observations))

            if joint > 0.0:
                running_sum += joint * math.log(joint / (marginal_a * marginal_b), 2)

    return running_sum

def count_matching_observations(match_dict, observations):
    matches = 0
    for o in observations:
        match = True
        for key in match_dict.keys():
            if o[key] != match_dict[key]:
                match = False
                break
        if match:
            matches += 1
    return matches

def save_mutual_information(nodes, observations, Vdata):
    edges_for_tree = []
    nodes_for_tree = nodes[:]

    for i in range(len(nodes_for_tree)):
        for j in range(i + 1, len(nodes_for_tree)):
            node_a = nodes_for_tree[i]
            node_b = nodes_for_tree[j]

            mi = manual_mutual_information(node_a, node_b, observations, bn.Vdata)
            edge = (node_a, node_b, mi)
            edges_for_tree.append(edge)

    dump_file = open('tanbc-mi.json', 'w')
    json.dump(edges_for_tree, dump_file)
    dump_file.close()
    print(edges_for_tree)

def load_mutual_information():
    dump_file = open('tanbc-mi.json', 'r')
    return json.load(dump_file)

def edges_for_maximum_spanning_tree(nodes, all_edges):
    added_nodes = []
    remaining_nodes = nodes[:]
    available_edges = all_edges[:]
    selected_edges = []

    # Select a random starting node.
    start_node = random.choice(remaining_nodes)
    remaining_nodes.remove(start_node)
    added_nodes.append(start_node)

    # Make all edge costs the negative of their original cost.
    available_edges = [[edge[0], edge[1], -edge[2]] for edge in available_edges]

    while len(remaining_nodes):
        next_edge = cheapest_tree_non_tree_edge(added_nodes, remaining_nodes, available_edges)
        selected_edges.append(next_edge)
        available_edges.remove(next_edge)

        if next_edge[0] in remaining_nodes:
            remaining_nodes.remove(next_edge[0])
            added_nodes.append(next_edge[0])

        if next_edge[1] in remaining_nodes:
            remaining_nodes.remove(next_edge[1])
            added_nodes.append(next_edge[1])

    directed_edges = assign_edge_directions(added_nodes, selected_edges)

    return directed_edges

def cheapest_tree_non_tree_edge(nodes_in_tree, nodes_not_in_tree, available_edges):
    valid_edges = []
    for edge in available_edges:
        if (edge[0] in nodes_in_tree and edge[1] in nodes_not_in_tree) or (edge[1] in nodes_in_tree and edge[0] in nodes_not_in_tree):
            valid_edges.append(edge)
    edge_costs = [edge[2] for edge in valid_edges]

    cheapest_edge = None
    cheapest_cost = float("inf")
    for edge in valid_edges:
        if edge[2] < cheapest_cost:
            cheapest_edge = edge
            cheapest_cost = edge[2]

    return cheapest_edge

def assign_edge_directions(nodes, edges):
    all_edges = edges[:]
    
    visited_nodes = []
    make_parent = []

    # Select random root node.
    root = random.choice(nodes)
    make_parent.append(root)

    # Visit each node.
    while len(make_parent) > 0:

        # Get next node to make a parent.
        visiting = make_parent[0]

        # Remove it from the make_parent list.
        make_parent = make_parent[1:]

        # Add it node to the visited list.
        visited_nodes.append(visiting)

        # Check each edge in all_edges.
        for i in range(len(all_edges)):

            # If current edge terminus is this node and if current edge start is not 
            # in the visited list, swap start and terminus.
            if all_edges[i][1] == visiting and all_edges[i][0] not in visited_nodes:
                all_edges[i][0], all_edges[i][1] = all_edges[i][1], all_edges[i][0]

            # If the terminus isn't in the list of visited nodes, add it to the list.
            if all_edges[i][0] == visiting and all_edges[i][1] not in make_parent:
                make_parent.append(all_edges[i][1])

    return all_edges

def remove_weights_from_edges(edges):
    removing_weights = edges[:]
    return [[edge[0], edge[1]] for edge in removing_weights]

def train_model(skeleton, observations):
    learner = PGMLearner()
    bn = learner.discrete_mle_estimateparams(skeleton, observations)
    return bn.V, bn.E, bn.Vdata

def test_model(observations, V, E, Vdata):
    correct = 0
    for observation in observations:
        map_class = classify_observation(observation, V, E, Vdata)
        if map_class == observation['poisonous']:
            correct += 1

    return correct, len(observations)

def classify_observation(observation, V, E, Vdata):
    o = observation.copy()
    remove_missing_data(o)
    remove_untrained_values(o, Vdata)

    if 'poisonous' not in o.keys():
        return None

    actual_class = o['poisonous']
    del o['poisonous']

    map_class = None
    map_prob = -1
    try:
        for value in Vdata['poisonous']['vals']:
            query = dict(poisonous=value)
            evidence = dict(o)
            factorization = pristine_fn(V, E, Vdata)
            result = factorization.specificquery(query, evidence)
            if result > map_prob:
                map_class = value
                map_prob = result
        return map_class
    except:
        return None

def remove_missing_data(observation):
    for key in observation.keys():
        if observation[key] == '?':
            del observation[key]

def remove_untrained_values(observation, learned_Vdata):
    for key in observation.keys():
        observation_value = observation[key]
        known_values = learned_Vdata[key]['vals']
        if observation_value not in known_values:
            del observation[key]

def k_fold_indices(k, n):
    testing_size = n / k
    training_size = n - testing_size

    validation_sets = []
    all_indices = range(n)    
    for i in range(k):
        training_indices = list(all_indices)
        index = testing_size * i
        testing_indices = range(index,index + testing_size)
        for j in testing_indices:
            training_indices.remove(j)
        validation_sets.append([training_indices, testing_indices])
    return validation_sets

def run_cross_validation(k, observations, graphSkeleton):
    cv_indices = k_fold_indices(k, len(observations))

    print('Performing {0}-fold cross validation.'.format(k))

    accuracies = []

    for i in range(len(cv_indices)):
        print('\tRunning iteration {0}.'.format(i + 1))
        testing = [observations[j] for j in cv_indices[i][1]]
        training = [observations[j] for j in cv_indices[i][0]]

        learned_V, learned_E, learned_Vdata = train_model(graphSkeleton, training)

        correct, n = test_model(testing, learned_V, learned_E, learned_Vdata)
        accuracy = correct/float(n)
        accuracies.append(accuracy)

        print('\t\tAccuracy: {0}'.format(accuracy))

    print('\tWeighted accuracy: {0}'.format(sum(accuracies)/len(accuracies)))

data_file_path = 'mushroom.csv'

# Create nodes from data file.
nodes = create_nodes_from_header(data_file_path)

# Start with no edges.
edges = []

# Parse observations from file.
observations = create_observations_from_csv(data_file_path, nodes)

# Create GraphSkeleton and learn parameters for disconnected network.
graphSkeleton = create_graph_skeleton(nodes, edges)
bn = PGMLearner().discrete_mle_estimateparams(graphSkeleton, observations)

# save_mutual_information(nodes[1:], observations, bn.Vdata)

# Load pre-calculated mutual information from file
edges_with_weights = load_mutual_information()

# Select edges and directions using Chow-Liu algorithm.
final_edges = edges_for_maximum_spanning_tree(nodes[1:], edges_with_weights)

# Strip unnecessary weights from edges.
edges = remove_weights_from_edges(final_edges)

# Add edges from class node to all other nodes.
for i in range(1, len(nodes)):
    edges.append([nodes[0], nodes[i]])

# Create a new GraphSkeleton with our tree-augmented network
graphSkeleton = create_graph_skeleton(nodes, edges)

# Run 10-fold cross validation.
run_cross_validation(10, observations, graphSkeleton)

# save_mutual_information(nodes[18:], observations, bn.Vdata)

# print(observations[0])
# print(count_matching_observations({'cap_shape': 'x'}, observations))