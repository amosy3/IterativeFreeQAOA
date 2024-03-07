"""Functions:
    - `obtain_noise_model`
    - `obtain_edges_maps`
    - `obtain_tqa_params`
Under the `if __name__ == "__main__"` section the actual data mining process occurs.    
"""

from typing import Iterable, Dict, Tuple, List

from qiskit_aer import AerSimulator
import pandas as pd
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from mine_data_2024_exp import run_exp, graphs_data_to_nx_graphs


def obtain_noise_model(num_qubits: int) -> NoiseModel:
    """Defines a simple noise model for a 'quantum computer' with `num_qubits` qubits,
    according to the following properties:
        - Readout errors - a unified readout error of 2% over all qubits.
        - Single-qubit gate errors - single-qubit gates (except RZ which is implemented in a noiseless
        manner) are defined under depolarizing noise channels with a depolarizing parameter of p=0.001.
        - Two-qubit gate errors - two-qubit gates (CNOT) are defined under depolarizing noise channels
        with a depolarizing parameter of p=0.01.
    """
    
    num_qubits = num_qubits
    noise_model = NoiseModel()

    # Defining gate errors
    depolarizing_error_1_qubit = depolarizing_error(0.001, 1)
    depolarizing_error_2_qubits = depolarizing_error(0.01, 2)

    # The untranspiled QAOA circuits in our implementation consists of the gates: X, SX, H, RX
    noise_model.add_all_qubit_quantum_error(depolarizing_error_1_qubit, ["x", "sx", "h", "rx"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error_2_qubits, ["cx"])

    # Defining the same readout error of 2% for all qubits
    uniform_readout_error = ReadoutError(
        [
            [0.98, 0.02],
            [0.02, 0.98]
        ]
    )

    for qubit_id in range(num_qubits):
        noise_model.add_readout_error(uniform_readout_error, [qubit_id])

    # Defining basis gates as in IBM's quantum computers, for transpilation if executed
    noise_model.add_basis_gates(["cx", "sx", "x", "rz"])

    return noise_model


def obtain_edges_maps(graphs_sizes: Iterable[int]) -> Dict[int, Dict[int, Tuple[int]]]:
    """Generates indexed maps of all possible edges in simple and connected graphs
    of each size defined as an integer in `graph_sizes`. E.g. the edge map of a 6-node graph is: {
        0: (0, 1), 1: (0, 2), 2: (0, 3), 3: (0, 4), 4: (0, 5), 5: (1, 2), 6: (1, 3), 7: (1, 4),
        8: (1, 5), 9: (2, 3),10: (2, 4), 11: (2, 5), 12: (3, 4), 13: (3, 5), 14: (4, 5)
    }
    """

    edges_maps = {}
    for num_nodes in graphs_sizes:
        
        edge_id = 0
        edges_map = {}
        for node_i in range(num_nodes):
            
            for node_j in range(node_i + 1, num_nodes):
                edges_map[edge_id] = (node_i, node_j)
                edge_id += 1
        
        edges_maps[num_nodes] = edges_map

    return edges_maps


def obtain_tqa_params(delta_t_values_file_path: str) -> Dict[int, List[float]]:
    """Computes parameters alpha_0, alpha_1, beta_0, beta_1 for the TQA method."""

    delta_t_data = pd.read_csv(delta_t_values_file_path)
    delta_t_values = delta_t_data["delta_t"]
    graphs_sizes = delta_t_data["num_nodes"]

    dts = {num_nodes: dt for num_nodes, dt in zip(graphs_sizes, delta_t_values)}

    params = {
        num_nodes: [
            0.75 * dts[num_nodes], 0.25 * dts[num_nodes], 0.25 * dts[num_nodes], 0.75 * dts[num_nodes]
        ] for num_nodes in graphs_sizes
    }

    return params


if __name__ == "__main__":

    graphs_sizes = [6, 7]

    # The number of "head" samples to take from the dataframes to sample 50 graphs
    # of the desired size.
    num_effective_graphs = {6: 61, 7: 55, 8: 56, 9: 52, 10: 50, 12: 50, 14: 50, 16: 50}
    
    graphs_data = {
        num_nodes: pd.read_csv(
            f"graphs_data/train_p_2_n_{num_nodes}_noise_None_graphs_random_prob_test" \
            f"_p_2_n_{num_nodes}_graph_random_prob_lb_0.3_ub_0.9_weighted_False_preds.csv"
        ) for num_nodes in graphs_sizes
    }

    edges_maps = obtain_edges_maps(graphs_sizes)
    tqa_params = obtain_tqa_params("delta_t_values.csv")
    methods = ["NN", "TQA"]

    # Executing experiment
    for num_nodes in graphs_sizes:
        
        for method in methods:

            # Sampling data of 50 graphs of size `num_nodes`
            cur_graphs_data = graphs_data[num_nodes].head(num_effective_graphs[num_nodes])

            # Transforming the graphs into nx.Graph objects
            graphs = graphs_data_to_nx_graphs(
                graphs_data=cur_graphs_data,
                edges_map=edges_maps[num_nodes],
                num_nodes=num_nodes
            )
                
            exp_name = f"num_nodes_{num_nodes}__p_2__method_{method}__" \
                        "simulation_CUSTOM_NOISE_MODEL_4000_shots__iters_0"
            
            # A hook intended to run user-defined TQA params
            if method == "TQA":
                method = tqa_params[num_nodes]
            
            print("**************")
            print("RUNNING", exp_name)
            print("**************")

            run_exp(
                graphs=graphs,
                raw_data=cur_graphs_data,
                backend=AerSimulator(noise_model=obtain_noise_model(num_nodes)),
                shots=4_000,
                meas=True,
                params_init_method=method,
                qaoa_layers=2,
                max_qaoa_iters=0,
                exp_name=exp_name,
                save_data_path="new_2024_exp_data"
            )

            print("**************")
            print("DONE", exp_name)
            print("**************")
                
        print("DONE ALL.")