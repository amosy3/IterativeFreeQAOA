import numpy as np
import pandas as pd
import random
import os
from Data_Generator_functions import *


import argparse



parser = argparse.ArgumentParser(description='Generate Erdos-Renyi graphs and store in '
                                             'csv file. If you give it the trained data-set, '
                                             'it as in the data-set. put --random_prob to generate graphs with random edge'
                                             'creation probablity, otherwise all edges with prob 0.5')
# TODO: add decimal list from csv file.

parser.add_argument("-n", "--number_of_nodes", help = "number of nodes of the graph = number of qubits", type=int)
parser.add_argument("-p", "--number_of_layers", help = "number of layers in the quantum circuit", type=int)
parser.add_argument("-file","--dataset_file", help = "data-set file which trains the NN, leave "
                                                     "empty to generate random graphs that some "
                                                     "have been trained on the NN", type = str, default='NoFile')
parser.add_argument("-prob","--edge_probability", help = "probablity of an edge to exist, relevant only to ER graphs, has no meaning if you use '--random_prob'", type = float,default=0.5)
parser.add_argument("-weighted","--weighted_graph",help="for generating weughted graph, write -weighted, works only for Erdos_Renyi graphs. don't put bounds with weighted graphs",default=False,action='store_true')
parser.add_argument("-data_size","--dataset_size", help = "number of graphs to generate",type=int, default=250)
parser.add_argument("--random_prob", help="creates random probablity Erdos-Renyi edge creation, write only '--random_prob' to generation graphs with random prob", default=False, action='store_true')
parser.add_argument("-lb_prob","--lower_bound_prob",
                    help= "minimum edge creation probability. Has no meaning if random_prob is False", default=0.3, type=float)
parser.add_argument("-ub_prob","--upper_bound_prob",
                    help= "maximum edge creation probability. Has no meaning if random_prob is False", default=0.9, type=float)


args = parser.parse_args()



print("n = ", args.number_of_nodes)
print("p = ", args.number_of_layers)
print("with probability of an edge: ", args.edge_probability)
print("weighted graph: ", args.weighted_graph)
print("dataset_size = ", args.dataset_size)
print("file: ", args.dataset_file)
print("random prob: ", args.random_prob)
print("lower bound prob: ", args.lower_bound_prob)
print("upper bound prob: ", args.upper_bound_prob)



nodes_num = args.number_of_nodes
p = args.number_of_layers  # Layer number
prob = args.edge_probability
weighted_bool = args.weighted_graph
file_str = args.dataset_file
graphs_num = args.dataset_size
random_prob_bool = args.random_prob
lower_bound_prob = args.lower_bound_prob
upper_bound_prob = args.upper_bound_prob




assert 0 <= prob <= 1, '-prob must be between 0 and 1'
assert 0 <= lower_bound_prob <= 1, '-lb_prob must be between 0 and 1'
assert 0 <= upper_bound_prob <= 1, '-ub_prob must be between 0 and 1'

num_possible_edges = int(nodes_num * (nodes_num - 1) / 2)


if weighted_bool:
    if os.path.isfile(file_str):
        trained_graphs = np.genfromtxt(file_str, delimiter=',')[1:, 1:(num_possible_edges + 1)]
        trained_graphs_pd = pd.read_csv(file_str)
        assert np.all(0.0 <= trained_graphs) and np.all(trained_graphs <= 1.0), 'input file is not as expected'
        assert np.any(np.logical_and(trained_graphs < 1.0, trained_graphs > 0.0)), 'you gave no weighted file'
        assert num_possible_edges == trained_graphs.shape[1], 'number of nodes and file do not agree'
        assert 'x' + str(num_possible_edges - 1) in trained_graphs_pd, 'number of nodes and file do not agree'
        assert 'x' + str(num_possible_edges) not in trained_graphs_pd, 'number of nodes and file do not agree'
    else:
        trained_graphs = np.zeros((1, num_possible_edges))
        print("you gave no file or not corrected path")
else:
    if os.path.isfile(file_str):
        trained_graphs = np.genfromtxt(file_str, delimiter=',')[1:, 1:(num_possible_edges + 1)]
        trained_graphs_pd = pd.read_csv(file_str)
        assert ((trained_graphs == 0) | (trained_graphs == 1)).all(), 'input file is not as expected'
        assert num_possible_edges == trained_graphs.shape[1], 'number of nodes and file do not agree'
        assert 'x'+str(num_possible_edges - 1) in trained_graphs_pd, 'number of nodes and file do not agree'
        assert 'x' + str(num_possible_edges) not in trained_graphs_pd, 'number of nodes and file do not agree'
    else:
        trained_graphs = np.zeros((1, num_possible_edges))
        print("you gave no file or not corrected path")


SavedGraps = np.empty([0, num_possible_edges + 1])
vec_column = ['x%s'%i for i in range(num_possible_edges)] + ['MaxCut']

file_name = "test_datasets/p_%s_n_%s_graph.csv" % (p, nodes_num)

if random_prob_bool:
    file_name = "test_datasets/p_%s_n_%s_graph_random_prob_lb_%s_ub_%s_weighted_%s.csv" % (p, nodes_num,lower_bound_prob,upper_bound_prob,weighted_bool)
else:
    file_name = "test_datasets/p_%s_n_%s_graph_const_prob_%s_weighted_%s.csv" % (p, nodes_num, prob, weighted_bool)
for i in range(graphs_num):

    vector_binars = generate_vector_weighted_binars(
        Data_Mat=np.vstack((trained_graphs,SavedGraps[:,:-1])), edge_prob=prob, weighted_graph_bool=weighted_bool, num_possible_edges=num_possible_edges,
        random_prob_bool=random_prob_bool, lower_bound_prob=lower_bound_prob, upper_bound_prob=upper_bound_prob
    )

    vector_data = np.array(vector_binars)

    Edges_possible_list = edge_list(nodes_num)
    Confugure_edges = Reduce2SampledConfiguaration(Edges_possible_list, vector_binars, weighted_bool)

    # calculate graphs maxcut
    Graph_maxcut = give_MaxCut_value(nodes_num, Confugure_edges, weighted_bool)
    vector_data = np.append(vector_data, Graph_maxcut)

    SavedGraps = np.vstack((SavedGraps, vector_data))



Data_to_save = pd.DataFrame(SavedGraps, columns=vec_column).to_csv(file_name)


