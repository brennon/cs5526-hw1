import csv, math

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

data_file_path = 'mushroom.csv'

# Create nodes from data file.
nodes = create_nodes_from_header(data_file_path)

# Remove class node.
nodes = nodes[1:]

# Start with no edges.
edges = []

# Parse observations from file.
observations = create_observations_from_csv(data_file_path, nodes)

# Create GraphSkeleton and learn parameters for disconnected network.
graphSkeleton = create_graph_skeleton(nodes, edges)
bn = PGMLearner().discrete_mle_estimateparams(graphSkeleton, observations)

# pprint(bn.Vdata)

factorization = pristine_fn(bn.V, bn.E, bn.Vdata)
result = factorization.specificquery(dict(cap_surface=['k'], population=['c']),dict())
print(result)