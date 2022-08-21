import networkx as nx
import numpy as np
# import cupy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
# from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.quantum_info import Pauli
# from qiskit.aqua.operators import op_converter
# from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.visualization import plot_histogram
from noise_functions import *


# backend = Aer.get_backend('statevector_simulator')
# # backend = Aer.get_backend('aer_simulator_statevector_gpu')
# backend.set_options(device='GPU')


def edge_list(nodes_num):
    """return all possible edge between nodes,
        where nodes_num is the number of nodes.
    """
    EdgeList = []
    for i in range(nodes_num):
        for j in range(nodes_num):
            if j > i:
                EdgeList.append((i, j))
    return EdgeList


def graph_to_adjacency_vector(graph):
    """take a graph and return its Adjacency Matrix as vector,
        only the off diagonal terms, for example:
        ([[0, x1, x2, x3],
          [x1, 0, x4, x5],
          [x2, x4, 0, x6],
          [x3, x5, x6, 0]])
    """
    Adjacency_Matrix = nx.adjacency_matrix(graph).todense()
    row, col = Adjacency_Matrix.shape
    vector_adjacencies = []
    for r in range(row):
        for c in range(col):
            if c > r:
                vector_adjacencies.append(Adjacency_Matrix[r, c])
    return vector_adjacencies


def list_of_binaries_from_decimal(num, Length_list):
    """take a number and return list of its binary in a list.
        add zeros to reach similar length
    """
    num_string = bin(int(num))[2:]
    if len(num_string) >= Length_list:
        return [int(char) for char in num_string]
    else:
        Add_Zeros_Num = Length_list - len(num_string)
        num_string = Add_Zeros_Num * '0' + num_string
        return [int(char) for char in num_string]


def Reduce2SampledConfiguaration(edges_list, edge_exist_list, weighted_bool):
    """take all possible edges.
        return every edge that their elements greater than 0 and their weight in tuple (u,v,weight)
         for weighted_bool = False return just (u,v)
    """
    assert (len(edges_list) == len(edge_exist_list))
    configuration = []
    if weighted_bool:
        for binar in range(len(edge_exist_list)):
            if edge_exist_list[binar] > 0.0:
                configuration.append(edges_list[binar] + (edge_exist_list[binar],))
    else:
        for binar in range(len(edge_exist_list)):
            if edge_exist_list[binar] == 1:
                configuration.append(edges_list[binar])

    return configuration


def list_of_binaries_Erdos_Renyi(prob, num_possible_edges, weighted_graph_bool: bool = False):
    """
    :param prob: probability of each edge to exist, otherwise it is deleted from graph.
    :param num_possible_edges: number of possible edges (when all nodes are connected).
    :return: binary list that the non existing edges are zeros. And the decimal value of the string.
    """
    if weighted_graph_bool:
        vector_binars = [np.random.rand() if np.random.rand() < prob else 0 for i in range(num_possible_edges)]
    else:
        vector_binars = [1 if np.random.rand() < prob else 0 for i in range(num_possible_edges)]
    return vector_binars

def decimal_number_fron_vector_binars(vector_binars):
    """
    :param vector_binars: list of 0s and 1s
    :return: the decimal number that represents the binary list
    """
    vector_binars_string = [str(n) for n in vector_binars]
    to_string = ''.join(vector_binars_string)
    return int(to_string,2)


def generate_vector_weighted_binars(Data_Mat: np.ndarray, edge_prob: float, weighted_graph_bool: bool, num_possible_edges: int,
                          random_prob_bool: bool, lower_bound_prob: float, upper_bound_prob: float):
    """
    :param Data_Mat: contains the previuos graphs
    :param edge_prob: edge probability creation for ER
    :param weighted_graph_bool: whether to put ones or weighted uniformly [0,1]
    :param num_possible_edges: number of possible edges
    :param random_prob_bool: whether to change randomly edge_prob between lower_bound_prob to upper_bound_prob
    :param lower_bound_prob: lower bound edge_prob
    :param upper_bound_prob: upper bound edge_prob
    :return: edge weights [0, 1, 0, 1,...] or for weighted [0.72,0.45,...]
    """
    if weighted_graph_bool:
        if random_prob_bool:
            # generate random prob
            rand_prob = random.uniform(lower_bound_prob, upper_bound_prob)
            # binary representaion
            vector_binars = list_of_binaries_Erdos_Renyi(rand_prob, num_possible_edges, weighted_graph_bool)
            # do not repeat on save configuration
            vector_binars = DontRepeatSameGraph(vector_binars, Data_Mat, rand_prob, weighted_graph_bool)
        else:
            # all graphs have uniform prob to edge creation
            # binary representaion
            vector_binars = list_of_binaries_Erdos_Renyi(edge_prob, num_possible_edges, weighted_graph_bool)
            # do not repeat on save configuration
            vector_binars = DontRepeatSameGraph(vector_binars, Data_Mat, edge_prob, weighted_graph_bool)
    else:
        if random_prob_bool:
            # generate random prob
            rand_prob = random.uniform(lower_bound_prob, upper_bound_prob)
            # binary representaion
            vector_binars = list_of_binaries_Erdos_Renyi(rand_prob, num_possible_edges)
            # do not repeat on save configuration
            vector_binars = DontRepeatSameGraph(vector_binars, Data_Mat, rand_prob)
        else:
            # all graphs have uniform prob to edge creation
            # binary representaion
            vector_binars = list_of_binaries_Erdos_Renyi(edge_prob, num_possible_edges)
            # do not repeat on save configuration
            vector_binars = DontRepeatSameGraph(vector_binars, Data_Mat, edge_prob)
    return vector_binars



def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    #qc.rz(2 * gamma, q2)
    qc.rz(- gamma, q2)
    qc.cx(q1, q2)


def get_cost_operator_circuit(qc, G, gamma):
    if nx.is_weighted(G):
        for i, j in G.edges():
            append_zz_term(qc, i, j, gamma * G[i][j].get('weight', 1.0))
    else:
        for i, j in G.edges():
            append_zz_term(qc, i, j, gamma)
    # return qc


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc






def state_num2str(basis_state_as_num, nqubits):
    return '{0:b}'.format(basis_state_as_num).zfill(nqubits)


def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)


def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2 ** nqubits, dtype=complex)
    for basis_state in range(2 ** nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state


def get_qaoa_circuit_sv(G, beta, gamma):
    assert (len(beta) == len(gamma))
    p = len(beta)  # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = QuantumCircuit(N)
    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        # qc = qc.compose(get_cost_operator_circuit(G, gamma[i]))
        get_cost_operator_circuit(qc, G, gamma[i])
        # qc = qc.compose(get_mixer_operator_circuit(G, beta[i]))
        qc.rx(2 * beta[i],range(N))
    # no measurement in the end!
    return qc


def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = '0{}b'.format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real ** 2 + val.imag ** 2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def compute_maxcut_energy_sv(sv, G):
    """Compute objective from statevector
    For large number of qubits, this is slow.
    """
    counts = state_to_ampl_counts(sv)
    return sum(maxcut_obj(np.array([int(x) for x in k]), G) * (np.abs(v) ** 2) for k, v in counts.items())


# compute_maxcut_energy_sv(sv, G)

def get_black_box_objective_sv(G, p):
    backend = Aer.get_backend('statevector_simulator')
    # backend = Aer.get_backend('aer_simulator_statevector_gpu')
    # backend.set_options(device='GPU')
    def f(theta):
        # let's assume first half is betas, second half is gammas
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit_sv(G, beta, gamma)
        sv = backend.run(qc.reverse_bits()).result().get_counts()
        # return the energy
        return compute_maxcut_energy(sv, G)

    return f

def maxcut_obj_weighted(x, G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= G[i][j]['weight']
    return cut


def maxcut_obj(x, G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut

def compute_maxcut_energy(counts, G, eps=1e-15):
    energy = 0
    if nx.is_weighted(G):
        for meas, meas_count in counts.items():
            if meas_count > eps:
                obj_for_meas = maxcut_obj_weighted(meas, G)
                energy += obj_for_meas * meas_count
    else:
        for meas, meas_count in counts.items():
            if meas_count > eps:
                obj_for_meas = maxcut_obj(meas, G)
                energy += obj_for_meas * meas_count
    return energy


def DontRepeatSameGraph(vector_binars,Data_Mat,prob, weighted_graph_bool: bool = False):
    """
    make sure for ot overlapping the same graphs.
    :param vector_binars: list of binaries that represent the graph's edges.
    :param Data_Mat: save the knowledge about each graph optimization
    :param num_possible_edges: number of maximum possible edges in a graph.
    :param prob: probability of each edge to exist in erdos-renyi graph.
    :return: vector_binars that does not appear in Data_Mat
    """
    num_possible_edges = len(vector_binars)
    if Data_Mat.shape[0] == 0:
        return vector_binars
    else:
        while vector_binars in Data_Mat[:,:num_possible_edges].tolist():
            vector_binars = list_of_binaries_Erdos_Renyi(prob, num_possible_edges, weighted_graph_bool)
        return vector_binars

def DontRepeatSameGraph2Datasets(vector_binars,Data_Mat1,Data_Mat2,prob):
    """
    make sure for ot overlapping the same graphs.
    :param vector_binars: list of binaries that represent the graph's edges.
    :param Data_Mat1: graphs that trained NN
    :param Data_Mat2: graphs from on going generation
    :param num_possible_edges: number of maximum possible edges in a graph.
    :param prob: probability of each edge to exist in erdos-renyi graph.
    :return: vector_binars that does not appear in Data_Mat
    """
    num_possible_edges = len(vector_binars)
    if Data_Mat2.shape[0] == 0:
        while vector_binars in Data_Mat1[:,:num_possible_edges].tolist():
            vector_binars = list_of_binaries_Erdos_Renyi(prob, num_possible_edges)
        return vector_binars
    else:
        while vector_binars in Data_Mat1[:,:num_possible_edges].tolist() or vector_binars in Data_Mat2[:,:num_possible_edges].tolist():
            vector_binars = list_of_binaries_Erdos_Renyi(prob, num_possible_edges)
        return vector_binars


def give_MaxCut_value(num_nodes, edge_list, weighted_bool):
    """return MaxCut value by running over
        all configurations.
    """
    num_calc_configurarion = 2 ** num_nodes
    maxcut_sol = 0
    if weighted_bool:
        assert len(edge_list[0]) == 3, 'edge_list not weighted or wrong elements'
        assert 0.0 <= edge_list[0][2] <= 1.0, 'weight not between 0 to 1'
        for conf_dec in range(1, num_calc_configurarion - 2):
            config_list = list_of_binaries_from_decimal(conf_dec, num_nodes)
            conf_cut = calc_maxcut_cut(config_list, edge_list,weighted_bool)
            if maxcut_sol < conf_cut:
                maxcut_sol = conf_cut
    else:
        for conf_dec in range(1, num_calc_configurarion - 2):
            config_list = list_of_binaries_from_decimal(conf_dec, num_nodes)
            conf_cut = calc_maxcut_cut(config_list, edge_list)
            if maxcut_sol < conf_cut:
                maxcut_sol = conf_cut
    return maxcut_sol

def calc_maxcut_cut(config_list,edge_list, weighted_bool: bool = False):
    """take configure list [0,1,1,0,...] and
        edge list [(0,2),(4,1),...].
        return its cut.
    """
    cut = 0
    if weighted_bool:
        for i, j, weight in edge_list:
            if config_list[i] != config_list[j]:
                # the edge is cut
                cut += weight
    else:
        for i, j in edge_list:
            if config_list[i] != config_list[j]:
                # the edge is cut
                cut += 1
    return cut

def add_edges_fun(G: nx.Graph,edges_configuratins_list , weighted_bool: bool = False) -> None:
    """
    Add edges to G graphs, determine if weighted graphs or standard graphs
    :param G: Graph
    :param edges_configuratins_list: list of edges and possible weights
    :param weighted_bool: whethet weighted or non weighted graph
    :return: None
    """
    if weighted_bool:
        G.add_weighted_edges_from(edges_configuratins_list)
    else:
        G.add_edges_from(edges_configuratins_list)



if __name__ == "__main__":
    edges_list = [(0,1,0.2),(0,2,0.5),(0,3,0.8),(1,2,0.5),(1,3,0.5),(2,3,0.8)]
    # edge_exist_list = [0,0.2,0,0.1,0,0.31]
    # # edge_exist_list = [0, 1, 0, 1, 0, 1]
    weighted_bool = True
    print(give_MaxCut_value(4, edges_list, weighted_bool))
    # a = Reduce2SampledConfiguaration(edges_list, edge_exist_list, weighted_bool)
    # print(a)
    # G = nx.Graph()
    # nodes_num = 4
    # G.add_nodes_from(range(nodes_num))
    # add_edges_fun(G=G,edges_configuratins_list = a, weighted_bool = weighted_bool)
    # nx.draw(G,with_labels=True)
    # plt.show()
    # G = nx.Graph()
    # nodes_num = 5
    # G.add_nodes_from(range(nodes_num))
    # G.add_edges_from([(0,1),(2,1), (0,4), (3,2)])
    # # qc = get_qaoa_circuit_sv(G,[0.1,0.2],[0.4,0.5])
    # # print(qc.draw())
    # # labels = nx.get_edge_attributes(G, 'weight')
    # # G.edges(data=True)
    # beta = np.array([0.2,0.5])
    # gamma = np.array([0.3,0.6])
    # abj = get_black_box_objective_sv(G, 2)
    # init_point = np.concatenate((beta,gamma))
    # res_sv = minimize(abj, init_point, method='BFGS', options={'maxiter': 1500, 'disp': True})