import csv

from libpgm.graphskeleton import GraphSkeleton
from libpgm.nodedata import NodeData
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner
from libpgm.tablecpdfactorization import TableCPDFactorization
from pprint import pprint

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

def train_model(skeleton, observations):
    learner = PGMLearner()
    bn = learner.discrete_mle_estimateparams(skeleton, observations)
    return bn.V, bn.E, bn.Vdata

def pristine_bn(V, E, Vdata):
    fresh_bn = DiscreteBayesianNetwork()
    fresh_bn.V = list(V)
    fresh_bn.E = list(E)
    fresh_bn.Vdata = Vdata.copy()
    return fresh_bn

def pristine_fn(V, E, Vdata):
    pristine = pristine_bn(V, E, Vdata)
    return TableCPDFactorization(pristine)

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

def sample_network_test():
    data_file_path = 'mushroom-short.csv'
    nodes = create_nodes_from_header(data_file_path)
    edges = create_edges_from_nodes(nodes)
    observations = create_observations_from_csv(data_file_path)
    graphSkeleton = create_graph_skeleton(nodes, edges)

    learned_V, learned_E, learned_Vdata = train_model(graphSkeleton, observations)

    for flu in learned_Vdata['flu']['vals']:

        fn = pristine_fn(learned_V, learned_E, learned_Vdata)
        print('P(flu={0}): {1}'.format(flu, fn.specificquery(dict(flu=flu), dict())))

        features = learned_Vdata.copy()
        del features['flu']

        for feature in features.keys():
            for feature_value in learned_Vdata[feature]['vals']:
                fn = pristine_fn(learned_V, learned_E, learned_Vdata)
                query = dict()
                query[feature] = feature_value
                print('\tP({0}={1}|flu={2}): {3}'.format(feature, feature_value, flu, fn.specificquery(query, dict(flu=flu))))

    fn = pristine_fn(learned_V, learned_E, learned_Vdata)
    result = fn.specificquery(dict(flu='t'), dict(chills='t',runny_nose='f',headache='m',fever='f'))
    print('P(flu=t|chills=t,runny_nose=f,headache=m,fever=n): {0}'.format(result))
    fn = pristine_fn(learned_V, learned_E, learned_Vdata)
    result = fn.specificquery(dict(flu='f'), dict(chills='t',runny_nose='f',headache='m',fever='f'))
    print('P(flu=f|chills=t,runny_nose=f,headache=m,fever=n): {0}'.format(result))

def test_splits(observations, graphSkeleton):
    n = len(observations)
    ratios = []
    accuracies = []
    training_testing_ratio = 0.3
    ratio_step = 0.3

    while True:
        training_n = int(n * training_testing_ratio)
        training = list(observations[0:training_n])
        testing = list(observations[training_n:])

        print('Number of training observations: {0}'.format(len(training)))
        print('Number of testing observations: {0}'.format(len(testing)))

        learned_V, learned_E, learned_Vdata = train_model(graphSkeleton, training)

        actual_classes = [testing[i]['poisonous'] for i in range(len(testing))]
        classification_results = []

        for test_observation in testing:
            remove_missing_data(test_observation)
            remove_untrained_values(test_observation, learned_Vdata)
            classification_result = classify_observation(test_observation, learned_V, learned_E, learned_Vdata)
            classification_results.append(classification_result)

        n_correct = sum([i[0] == i[1] for i in zip(actual_classes, classification_results)])
        accuracy = n_correct / float(len(actual_classes))

        ratios.append(training_testing_ratio)
        accuracies.append(accuracy)

        print('Accuracy: {0}'.format(accuracy))
        training_testing_ratio += ratio_step

        if training_testing_ratio >= 1:
            break

    import matplotlib.pyplot as plt
    plt.plot(accuracies, ratios)
    plt.show()

data_file_path = 'mushroom.csv'
nodes = create_nodes_from_header(data_file_path)
edges = create_edges_from_nodes(nodes)
observations = create_observations_from_csv(data_file_path, nodes)
graphSkeleton = create_graph_skeleton(nodes, edges)

print('Graph skeleton created with {0} nodes and {1} edges:'.format(len(graphSkeleton.V), len(graphSkeleton.E)))
for node in graphSkeleton.V:
    print('\tNode \'{0}\', parents({0}): {1}'.format(node, graphSkeleton.getparents(node)))

run_cross_validation(10, observations, graphSkeleton)