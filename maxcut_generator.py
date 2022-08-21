# in this code we changed the way we save data, instead of decimal number to vector of binary numbers that represent the
# adjancency matrix.
# We also changed the ZZ interaction like in Crooks article. Cx Rz(-\gamma) Cx. instead Cx Rz(2\gamma) Cx.



import networkx as nx
import numpy as np
# import cupy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, transpile
# from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.quantum_info import Pauli
# from qiskit.aqua.operators import op_converter
# from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.visualization import plot_histogram
from Data_Generator_functions import *
from noise_functions import *
from tqa_functions import *

from qiskit.providers.aer import AerSimulator

from qiskit.test.mock import FakeMumbai, FakeMontreal

import argparse


parser = argparse.ArgumentParser(description='Generate posible maxcut configuration and return their optimized parameters. \n'
                                             'choose method of optimizer and graph construction: Erdos-Renyi or decim to binary vector.\n'
                                             'Erdos-Renyi:each edge exist with certain probability. \n'
                                             'now we save all vector of the adjacency matrix instead of decimal number.\n'
                                             'And given parameters similar to Lukin Article.'
                                             'put --random_prob to generate graphs with random edge'
                                             'creation probablity, otherwise all edges with prob 0.5 or your given float.'
                                             'Added noisy device simulation, for that write "-noisy"'
                                             'Give you the ability to decide on initialization parameters. default TQA.')


parser.add_argument("-n", "--number_of_nodes", help = "number of nodes of the graph = number of qubits", type=int)
parser.add_argument("-p", "--number_of_layers", help = "number of layers in the quantum circuit", type=int)
parser.add_argument("--dataset_size", type=int, default=5000)
parser.add_argument("-init", "--init_params",help="init parameters linear method, 'TQA', 'Lukin1', 'Lukin2'(previous trained method) or 'half_pi'", default='TQA', type=str)
parser.add_argument("-b", "--with_bounds", help = "Give True to bound QAOA parameters between 0 to pi, otherwise the parameters are not bounded", type=bool, default=False)
parser.add_argument("-ER", "--Erdos_Renyi", help = "On default generate Erdos-Renyi graphs, only write '-ER' to generation from decimal to binary list", default=True, action='store_false')
parser.add_argument("-prob", "--edge_probability", help = "probablity of an edge to exist, relevant only to ER graphs", type = float,default=0.5)
parser.add_argument("-weighted","--weighted_graph",help="for generating weughted graph, write -weighted, works only for Erdos_Renyi graphs. don't put bounds with weighted graphs",default=False,action='store_true')
parser.add_argument("-optimizer", "--optimizer", help = "choose optimizer, options: 'COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'", type = str,default= 'BFGS')
parser.add_argument("-noisy", "--noisy_simulation",help="use noisy simulation,write only '-noisy'",default=False, action='store_true')
parser.add_argument("-device", "--simulator_name",help="quantum computer noisy simulator name, 'Mumbai' or 'Montreal'. relevant only if stated '-noisy'", default='Montreal',type=str)
parser.add_argument("--random_prob", help="creates random probablity Erdos-Renyi edge creation, write only '--random_prob' to generation graphs with random prob", default=False, action='store_true')
parser.add_argument("-lb_prob", "--lower_bound_prob",
                    help= "minimum edge creation probability. Has no meaning if random_prob is False", default=0.3, type=float)
parser.add_argument("-ub_prob", "--upper_bound_prob",
                    help= "maximum edge creation probability. Has no meaning if random_prob is False", default=0.9, type=float)


args = parser.parse_args()


print("n = ", args.number_of_nodes)
print("p = ", args.number_of_layers)
print("dataset_size = ", args.dataset_size)
print("init parameter method: ", args.init_params)
print("with bounds: ",args.with_bounds)
print("type of optimizer: ",args.optimizer)
if args.noisy_simulation:
    print("run noisy simulation")
    print("noise simulator device: ",args.simulator_name)
else:
    print("run non-noisy simulation")
print("Does generate Erdos-Renyi graphs: ", args.Erdos_Renyi)
if args.Erdos_Renyi:
    print("weighted graph: ", args.weighted_graph)
    if args.random_prob:
        print("random prob: ", args.random_prob)
        print("lower bound prob: ", args.lower_bound_prob)
        print("upper bound prob: ", args.upper_bound_prob)
    else:
        print("constant probability of an edge: ", args.edge_probability)
else:
    print("generate graph by converting decimal number to binary, \nedge exist with 1 and eliminated where 0")

if args.weighted_graph and args.with_bounds:
    print("YOU PUT WEIGHTED GRAPH AND BOUNDS ON THE PARAMETER SPACE, BETTER AVOID THAT")



nodes_num = args.number_of_nodes
p = args.number_of_layers  # Layer number
init_params = args.init_params
with_bounds = args.with_bounds
optimizer = args.optimizer
prob = args.edge_probability
ER_bool = args.Erdos_Renyi
random_prob_bool = args.random_prob
lower_bound_prob = args.lower_bound_prob
upper_bound_prob = args.upper_bound_prob
noisy_simulation_bool = args.noisy_simulation
noisy_sim_name = args.simulator_name if noisy_simulation_bool else 'noiseless'
weighted_bool = args.weighted_graph





assert 0 <= prob <= 1, '-prob must be between 0 and 1'
assert optimizer in ['COBYLA', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP'], '-optimizer need to be one of the proposed list in str'
assert 0 <= lower_bound_prob <= 1, '-lb_prob must be between 0 and 1'
assert 0 <= upper_bound_prob <= 1, '-ub_prob must be between 0 and 1'
assert noisy_sim_name in ['Montreal', 'Mumbai', 'noiseless'], "'-device' has wrong name"





num_possible_edges = int(nodes_num * (nodes_num - 1) / 2)
Max_decimal_num = (2 ** num_possible_edges) - 1



# Data Metrix is orgenized from row vectors
# as followed: [decimal,optimized cost,gamma*,beta*]
Data_Mat = np.empty([0, num_possible_edges + 1 + 2 * p])

assert args.dataset_size <= (Max_decimal_num + 1), '--dataset_size cannot be bigger than number of possible configurations'

vec_column = ['x%s'%i for i in range(num_possible_edges)]
columns = vec_column + ['cost'] + ['b%s'%i for i in range(p)] + ['g%s'%i for i in range(p)] #+ ["MaxCut"]

Bounds = [[ 0, np.pi]] * p + [[0, 2*np.pi]] * p

decimals_list = []

#noise properties
noise_device_backend = FakeMontreal() if noisy_sim_name == 'Montreal'  else FakeMumbai()


note_bounds = "_bounded_params" if args.with_bounds else ''
note_graph = "_ER" if ER_bool else "_from_decim"

#file name
graph_string = 'random_prob' if random_prob_bool else 'None'
noise_string = 'HW' if args.noisy_simulation else 'None'
weighted_string = 'graphs' if weighted_bool else 'None'

file_name = "train_datasets/p_%s_n_%s_noise_%s_graphs_%s_weighted_%s.csv" %(p, nodes_num, noise_string, graph_string, weighted_string)

if ER_bool:
    for i in range(args.dataset_size):

        vector_binars = generate_vector_weighted_binars(
            Data_Mat=Data_Mat, edge_prob=prob, weighted_graph_bool=weighted_bool, num_possible_edges=num_possible_edges,
            random_prob_bool=random_prob_bool, lower_bound_prob=lower_bound_prob, upper_bound_prob=upper_bound_prob
            )


        # save conifguration
        vector_data = np.array(vector_binars)

        # generate graph
        G = nx.Graph()
        G.add_nodes_from(range(nodes_num))

        # all possible edges and reduce to specific configuation
        Edges_possible_list = edge_list(nodes_num)
        Confugure_edges = Reduce2SampledConfiguaration(Edges_possible_list, vector_binars, weighted_bool)

        # add edges to graph
        add_edges_fun(G=G,edges_configuratins_list=Confugure_edges,weighted_bool=weighted_bool)
        # circuit and cost function, noisy or noiseless
        if noisy_simulation_bool:
            # initial parameters
            init_point = initial_point_params_noisy(method=init_params, p=p, G=G,device_backend = noise_device_backend)
            obj = get_black_box_objective_noisy(G, p,noise_device_backend)
        else:
            # initial parameters
            init_point = initial_point_params(method=init_params, p=p, G=G)
            obj = get_black_box_objective_sv(G, p)
        # get parameters as vector



        if with_bounds:
            res_sv = minimize(obj, init_point, method=optimizer, bounds=Bounds, options={'maxiter': 1500, 'disp': True})
        else:
            res_sv = minimize(obj, init_point, method=optimizer, options={'maxiter': 1500, 'disp': True})

        # save data
        vector_data = np.append(vector_data, res_sv['fun'])
        vector_data = np.append(vector_data, res_sv['x'])

        # calculate graphs maxcut

        Data_Mat = np.vstack((Data_Mat, vector_data))
        if i % 5 == 0:
            pd.DataFrame(Data_Mat, columns=columns).to_csv(file_name)
else:
    all_graphs = random.sample(range(1, Max_decimal_num + 1), args.dataset_size)
    for decim in all_graphs:

        # binary representaion
        vector_binars = list_of_binaries_from_decimal(decim, num_possible_edges)
        vector_data = np.array(vector_binars)

        # generate graph
        G = nx.Graph()
        G.add_nodes_from(range(nodes_num))

        # all possible edges and reduce to specific configuation
        Edges_possible_list = edge_list(nodes_num)
        Confugure_edges = Reduce2SampledConfiguaration(Edges_possible_list, vector_binars)

        # add edges to graph
        G.add_edges_from(Confugure_edges)
        # circuit and cost function, noisy or noiseless
        if noisy_simulation_bool:
            # initial parameters
            init_point = initial_point_params_noisy(method=init_params, p=p, G=G, device_backend=noise_device_backend)
            obj = get_black_box_objective_noisy(G, p, noise_device_backend)
        else:
            # initial parameters
            init_point = initial_point_params(method=init_params, p=p, G=G)
            obj = get_black_box_objective_sv(G, p)
        # get parameters as vector


        # make optimization
        if with_bounds:
            res_sv = minimize(obj, init_point, method=optimizer, bounds=Bounds, options={'maxiter': 1500, 'disp': True})
        else:
            res_sv = minimize(obj, init_point, method=optimizer, options={'maxiter': 1500, 'disp': True})

        # save data
        vector_data = np.append(vector_data, res_sv['fun'])
        vector_data = np.append(vector_data, res_sv['x'])

        # calculate graphs maxcut

        Data_Mat = np.vstack((Data_Mat, vector_data))
        if decim % 5 == 0:
            pd.DataFrame(Data_Mat, columns=columns).to_csv(file_name)

Data_to_save = pd.DataFrame(Data_Mat, columns=columns).to_csv(file_name)