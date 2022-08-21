import pandas as pd
import numpy as np
from GraphER import GraphER
import os
import argparse

parser = argparse.ArgumentParser(description='Generate Erdos-Renyi test dataset'
                                             'with the same properties as in train_datasets')

parser.add_argument("-n", "--number_of_nodes", help = "also determines number of qubits", type=int)
parser.add_argument("-p", "--number_of_layers", help = "number of layers in the quantum circuit", type=int)
parser.add_argument("-data","--dataset_size",help="number of graphs for testing", type=int, default=5000)
parser.add_argument("-prob", "--edge_probabilities", help = "generates probability for edge to exist in each graph with uniform probablity U(low,high)", type = tuple,default=(0.5,0.5))
parser.add_argument("-weighted","--weighted_graph",help="each edge with weights in U(0,1)",default=False,action='store_true')
parser.add_argument("-optimizer", "--optimizer", help = "options: 'COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'", type = str,default= 'BFGS')
parser.add_argument("-bounds", "--with_bounds", help="add bounds for parametrs, not to weighted graph",default=False, action='store_true')
parser.add_argument("-noisy", "--noisy_simulation",help="use noisy simulation,write only '-noisy'",default=False, action='store_true')
# TODO noisy simulation
args = parser.parse_args()

nodes_num = args.number_of_nodes
p = args.number_of_layers
dataset_size = args.dataset_size
prob = args.edge_probabilities
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

num_possible_edges = int(nodes_num * (nodes_num - 1) / 2)
adj_vec_cols = ['x(%s,%s)'%(i, j) for i in range(nodes_num) for j in range(1,nodes_num) if j>i]

read_file = "train_datasets/p_%s_n_%s_prob_%s_weighted_%s_bounds_%s_noisy_%s.csv"%\
            (p, nodes_num, prob, weighted_bool,bounds_bool, noisy_bool)

if os.path.isfile(read_file):
    train_df = pd.read_csv(read_file)
else:
    raise AssertionError("you gave no file or not corrected path")

file_name = "test_datasets/p_%s_n_%s_prob_%s_weighted_%s_bounds_%s_noisy_%s_test_data.csv" % \
            (p, nodes_num, prob, weighted_bool,bounds_bool, noisy_bool)

test_df = pd.DataFrame({}, columns=adj_vec_cols)

for i in range(dataset_size):
    # merged train and test DataFrames
    df_merged = pd.concat([train_df, test_df])
    vector_edges = GraphER.generate_vector_adjacency_elements(Data=df_merged,
                                                              edge_prob=prob,
                                                              weighted_bool=weighted_bool,
                                                              num_possible_edges=num_possible_edges,
                                                              adjacency_vec=adj_vec_cols)
    test_df.loc[len(test_df.index)] = vector_edges
    test_df.to_csv(file_name, index=False)





