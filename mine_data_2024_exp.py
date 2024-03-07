"""Functions: `graphs_data_to_nx_graphs`, `callback`, `run_exp`."""

import os
import json
from typing import Dict, Tuple, Optional, Union, List

import networkx as nx
import pandas as pd
from scipy.optimize import OptimizeResult

from qaoa_maxcut_second_ver import QAOAMaxCutSVSecondVer
from cut_funs import brute_force_maxcut


def graphs_data_to_nx_graphs(
    graphs_data: pd.DataFrame,
    edges_map: Dict[int, Tuple[int, int]],
    num_nodes: int
) -> Dict[int, nx.Graph]:
    """Transforming graph data from an edges_map format (see the `obtain_edges_map`
    function for details) into an nx.Graph object."""

    graphs = {}
    for graph_index in range(len(graphs_data)):
        graph = nx.Graph(
            [
                edges_map[i] for i in edges_map.keys() if graphs_data.loc[graph_index][i] == 1
            ]
        )

        if graph.number_of_nodes() != num_nodes:
            continue
            
        graphs[graph_index] = graph

    return graphs


def callback(intermediate_result: OptimizeResult) -> None:
    """Callback function to print intermediate result throughout the optimization process."""

    print("CALLBACK,", intermediate_result)

def run_exp(
    graphs: Union[Dict[int, nx.Graph], List[nx.Graph]],
    raw_data: pd.DataFrame,
    backend,
    shots: int,
    meas: bool,
    params_init_method: str,
    qaoa_layers: int,
    max_qaoa_iters: int,
    exp_name: str,
    save_data_path: str,
    opt_precision: Optional[float] = None,
    transpile_circuits: Optional[bool] = False
) -> float:
    """Executes QAOA for graphs in `graphs` on `backend` with `shots` shots. Saves and documents
    all desired data and metadata into `save_data_path`. Returns the average approximation ratio
    achieved by QAOA averaged all graphs in `graphs`."""

    num_graphs = len(graphs)

    total_appx_ratio = 0
    gen_data = pd.DataFrame(
        {
            "graph_index": [],
            "qaoa_cut": [],
            "maxcut": [],
            "appx_ratio": []
        }
    )

    for graph_index, graph in graphs.items():
        
        qaoa_sv = QAOAMaxCutSVSecondVer(
            graph,
            reps=qaoa_layers,
            backend=backend,
            shots=shots,
            meas=meas,
            transpile_circuits=transpile_circuits
        )

        if params_init_method == "NN":
            init_params = raw_data.loc[graph_index].loc[["b0", "b1", "g0", "g1"]]
            qaoa_sv.determine_initial_point(init_params)
        elif params_init_method == "TQA":
            qaoa_sv.determine_initial_point()
        elif isinstance(params_init_method, list): # A hook to run TQA with chosen params
            qaoa_sv.determine_initial_point(params_init_method)
        else:
            raise Exception("Invalid input. `params_init_method` can take only 'NN' or 'TQA' values.")

        optimizer = "BFGS"
        print("Optimizer =", optimizer)

        qaoa_cut = -qaoa_sv.optimize(
            maxiter=max_qaoa_iters,
            optimizer=optimizer,
            callback=callback,
            tol=opt_precision,
        ).fun
        maxcut = brute_force_maxcut(graph).best_score
        cur_appx_ratio = qaoa_cut / maxcut

        gen_data = pd.concat(
            [
                gen_data,
                pd.DataFrame(
                    {
                        "graph_index": [graph_index],
                        "qaoa_cut": [qaoa_cut],
                        "maxcut": [maxcut],
                        "appx_ratio": [cur_appx_ratio]
                    }
                )
            ]
        )
        total_appx_ratio += cur_appx_ratio

    avg_appx_ratio = total_appx_ratio / num_graphs
    if isinstance(backend.name, str):
        backend_name = backend.name
    else:
        backend_name = backend.name()

    metadata = {
        "num_graphs": num_graphs,
        "avg_appx_ratio": avg_appx_ratio,
        "backend": backend_name,
        "shots": shots,
        "qaoa_layers": qaoa_layers,
        "max_qaoa_iters": max_qaoa_iters,
        "valid_graphs_indexes": list(graphs.keys())
    }

    save_data_path = f"{save_data_path}/{exp_name}"
    os.mkdir(save_data_path)

    raw_data.to_csv(f"{save_data_path}/raw_data.csv")
    gen_data.to_csv(f"{save_data_path}/exp_results_data.csv")

    with open(f"{save_data_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return avg_appx_ratio