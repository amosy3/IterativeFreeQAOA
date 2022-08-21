import pandas as pd
import networkx as nx
import numpy as np
# from qiskit.providers.fake_provider import FakeMontreal
from GraphER import GraphER
from QAOAMaxCutSV import QAOAMaxCutSV
import argparse

parser = argparse.ArgumentParser(description= "Generates random or const Erdos-Ranyi graphs "
                                              "and optimize QAOA for the MaxCut problem. "
                                              "Saves graphs and corresponding QAOA's parameters")


parser.add_argument("-n", "--number_of_nodes", help = "also determines number of qubits", type=int)
parser.add_argument("-p", "--number_of_layers", help = "number of layers in the quantum circuit", type=int)
parser.add_argument("-data","--dataset_size",help="number of graphs to collect data", type=int, default=5000)
parser.add_argument("-prob", "--edge_probabilities", help = "generates probability for edge to exist in each graph with uniform probablity U(low,high), write: 0.3 0.9", nargs='+', type=float, default=(0.5,0.5))
parser.add_argument("-weighted","--weighted_graph",help="each edge with weights in U(0,1)",default=False,action='store_true')
parser.add_argument("-optimizer", "--optimizer", help = "options: 'COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'", type = str,default= 'BFGS')
parser.add_argument("-bounds", "--with_bounds", help="add bounds for parametrs, not to weighted graph",default=False, action='store_true')
parser.add_argument("-noisy", "--noisy_simulation",help="use noisy simulation,write only '-noisy'",default=False, action='store_true')
# TODO noisy simulation
args = parser.parse_args()

nodes_num = args.number_of_nodes
p = args.number_of_layers
dataset_size = args.dataset_size
prob = tuple(args.edge_probabilities)
weighted_bool = args.weighted_graph
optimizer = args.optimizer
bounds_bool = args.with_bounds
noisy_bool = args.noisy_simulation



print("n = ", nodes_num)
print("p = ", p)
print("dataset_size = ", dataset_size)
print("edge probability: ", prob)
print("weighted graph: ", weighted_bool)
print("type of optimizer: ",optimizer)
print("add bounds: ", bounds_bool)
print("Noisy: ", noisy_bool)

# noise_device_backend = FakeMontreal()

assert (0.0 <= prob[0] <= 1.0) and (0.0 <= prob[1] <= 1.0) and (prob[0] <= prob[1]), '-probabilities must be between 0 and 1'
assert optimizer in ['COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'], '-optimizer need to be one of the proposed list in str'


num_possible_edges = int(nodes_num * (nodes_num - 1) / 2)
bounds = [(0, np.pi)] * p + [(0, 2*np.pi)] * p

assert dataset_size <= 2**num_possible_edges, "data cannot be more than possible configurations"


adj_vec_cols = ['x(%s,%s)'%(i, j) for i in range(nodes_num) for j in range(1,nodes_num) if j>i]
columns = adj_vec_cols + ['beta%s'%i for i in range(p)] + ['gamma%s'%i for i in range(p)]

Data_collector = pd.DataFrame({}, columns=columns)
file_name = "train_datasets/p_%s_n_%s_prob_%s_weighted_%s_bounds_%s_noisy_%s.csv"%\
            (p, nodes_num, prob, weighted_bool,bounds_bool, noisy_bool)

for i in range(dataset_size):
    vector_edges = GraphER.generate_vector_adjacency_elements(Data=Data_collector,
                                                      edge_prob=prob,
                                                      weighted_bool=weighted_bool,
                                                      num_possible_edges=num_possible_edges,
                                                      adjacency_vec=adj_vec_cols)

    # generate graph
    G = GraphER(vec_adjacancy=vector_edges,
                num_nodes=nodes_num)


    qaoa_graph = QAOAMaxCutSV(Graph=G,
                              reps=p)
    # initial parameters, adding None results full TQA protocol
    qaoa_graph.determine_initial_point()

    # optimization result
    if bounds_bool and not nx.is_weighted(G):
        opt_sv_result = qaoa_graph.optimize(optimizer=optimizer,
                                            bounds=bounds)
    else:
        opt_sv_result = qaoa_graph.optimize(optimizer=optimizer)

    # combining graphs to optitimizing params
    graph_data = np.concatenate((vector_edges,opt_sv_result.x))
    Data_collector.loc[len(Data_collector.index)] = graph_data
    Data_collector.to_csv(file_name, index=False)














