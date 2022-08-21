import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Sequence, Optional






class GraphER(nx.Graph):
    """
    generates nx graph from vector of adjacancy matrix.
    """
    def __init__(self,
                 vec_adjacancy: np.ndarray,
                 num_nodes: int
                 ) -> nx.Graph:
        super().__init__()
        self._vec_adj = vec_adjacancy if type(vec_adjacancy) == np.ndarray else np.array(vec_adjacancy)
        self._n = num_nodes
        self._weighted_bool = not np.array_equal(self._vec_adj, self._vec_adj.astype(bool))
        self._poss_edges = self._edge_list()
        self._edges_list = self._graph_edges_n_weights()
        self.add_nodes_from(range(self._n))
        if self._weighted_bool:
            self.add_weighted_edges_from(self._edges_list)
        else:
            self.add_edges_from(self._edges_list)


    def _edge_list(self) -> List[Tuple]:
        """return all possible edge between nodes,
            where nodes_num is the number of nodes.
        """
        EdgeList = []
        for i in range(self._n):
            for j in range(self._n):
                if j > i:
                    EdgeList.append((i, j))
        return EdgeList

    def _graph_edges_n_weights(self) -> List[Tuple]:
        """
        in order to generate graph
        :return: list of edges and weights for weighted graphs
        """
        assert len(self._vec_adj) == len(self._poss_edges)
        # lisf of graph edges
        configuration = []
        if self._weighted_bool:
            for edge in range(len(self._vec_adj)):
                if self._vec_adj[edge] > 0.0:
                    configuration.append(self._poss_edges[edge] + (self._vec_adj[edge],))
        else:
            for edge in range(len(self._vec_adj)):
                if self._vec_adj[edge] > 0.0:
                    configuration.append(self._poss_edges[edge])
        return configuration

    @property
    def possible_edges(self):
        return self._poss_edges

    @staticmethod
    def generate_vector_adjacency_elements(Data: pd.DataFrame,
                                           edge_prob: Tuple,
                                           weighted_bool: bool,
                                           num_possible_edges: int,
                                           adjacency_vec: List[str]
                                           ) -> np.ndarray:
        """
        vector represents adjacancy matrix, e.g. [(0,1),(0,2),(1,2)]
        :param Data: data of previous optimized graphs
        :param edge_prob: edge probability creation (low,high)
        :param weighted_bool: True for weighted graphs
        :param num_possible_edges: possible edges in graph
        :param adjacency_vec: name list of each potential edge e.g. [x(0,1),x(0,2),...,x(n-1,n)]
        :return: vector of edges to create a graph. with or without weights.
        """
        previous_graphs = Data[adjacency_vec].values
        # generates random edge creation probability
        graph_edge_prob = np.random.uniform(low=edge_prob[0], high=edge_prob[1])
        if weighted_bool:
            # generates weighted vector represents graph adjecancy matrix
            vector_edges = np.array([np.random.rand() if np.random.rand() < graph_edge_prob else 0 for i in
                                     range(num_possible_edges)])
            while any(np.equal(previous_graphs, vector_edges).all(1)) or np.sum(vector_edges) == 0.0:
                vector_edges = np.array([np.random.rand() if np.random.rand() < graph_edge_prob else 0 for i in
                                         range(num_possible_edges)])
        else:

            vector_edges = np.array([1.0 if np.random.rand() < graph_edge_prob else 0.0 for i in
                                     range(num_possible_edges)])
            while any(np.equal(previous_graphs, vector_edges).all(1)) or np.sum(vector_edges) == 0.0:
                vector_edges = np.array([1.0 if np.random.rand() < graph_edge_prob else 0.0 for i in
                                         range(num_possible_edges)])
        return vector_edges








if __name__ == '__main__':
    n = 4
    a = GraphER([0,0.2,0,0.1,0,0.31], n)
    b = GraphER([0,1,0,1,0,1], n)
    print(a)