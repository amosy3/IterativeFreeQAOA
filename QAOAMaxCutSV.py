from GraphER import GraphER
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Tuple, Dict, Sequence, Optional, Callable
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
import itertools
from scipy.optimize import minimize
import copy



class QAOAMaxCutSV:
    """
    create QAOA circuit with state vector form which solves the
    maximum cut problem.
    First initialize paramters, default TQA protocol.
    Then use optimize.
    optimizer from scipy.
    """

    BACKEND = Aer.get_backend('statevector_simulator')

    def __init__(self,
                 Graph: GraphER = None,
                 reps: int = 1
                 ) -> None:
        self._G = Graph
        self.p = reps
        self._qaoa_circ = self._get_qaoa_circuit_sv()
        self._counter = itertools.count()



    def optimize(self,
                 optimizer: str = 'BFGS',
                 bounds: Optional[Sequence[Tuple[float,float]]] = None,
                 maxiter: int = 1000,
                 tol: Optional[float] = None,
                 callback: Optional[Callable] = None
                 ) -> Dict:
        try:
            self._init_params
        except AttributeError:
            raise AttributeError('run method determine_initial_point to '
                                 'determine initial point.')

        obj = lambda theta: self._black_box_obj(params = theta)
        result = minimize(fun=obj, x0=copy.copy(self._init_params),
                          method=optimizer, bounds=bounds, tol=tol,
                          callback=callback,
                          options={'maxiter': maxiter, 'disp': True})
        return result



    def _get_qaoa_circuit_sv(self) -> QuantumCircuit:
        """
        build QAOA circuit
        """
        #  obtain hyperparameters
        self._beta = ParameterVector('ß', self.p)
        self._gamma = ParameterVector('γ', self.p)
        N = self._G.number_of_nodes()
        qc = QuantumCircuit(N)
        # initialize Hadamards
        qc.h(range(N))
        for i in range(self.p):
            # cost operator
            self._cost_operator_circuit(qc, self._gamma[i])
            # mixer
            qc.rx(2 * self._beta[i], range(N))
        # no measurement in the end!
        return qc

    def _qaoa_cost(self,
                   beta: np.ndarray,
                   gamma: np.ndarray
                   ) -> float:
        """
        run qaoa by assigning (beta,gamma) values
        :return: qaoa cost <H_c>
        """
        _assigned_circ = self._qaoa_circ.assign_parameters(
            {self._beta: beta, self._gamma: gamma})
        # state vector results
        sv = self.BACKEND.run(_assigned_circ.reverse_bits()).result().get_counts()
        return self.compute_maxcut_energy(sv)

    def compute_maxcut_energy(self,
                              counts: Dict,
                              eps: int =1e-15
                              ) -> float:
        energy = 0
        for meas, meas_count in counts.items():
            if meas_count > eps:
                obj_for_meas = self._maxcut_obj(meas)
                energy += obj_for_meas * meas_count
        return energy

    def _maxcut_obj(self,
                    x: str
                    ) -> float:
        cut = 0
        for i, j in self._G.edges():
            if x[i] != x[j]:
                # the edge is cut
                cut -= self._G[i][j].get('weight', 1.0)
        return cut

    def _append_zz_term(self, qc: QuantumCircuit,
                        q1: int, q2: int,
                        gamma
                        ) -> None:
        """
        append operator e^{-γ/2 ZZ} for each edge in graph
        """
        qc.cx(q1, q2)
        qc.rz(- gamma, q2)
        qc.cx(q1, q2)

    def _cost_operator_circuit(self,
                               qc: QuantumCircuit,
                               gamma
                               ) -> None:
        """
        append to qc (QuantumCircuit) cost operator:
        e^{-iγH_c}
        """
        for i, j in self._G.edges():
            self._append_zz_term(qc, i, j, gamma * self._G[i][j].get('weight', 1.0))



    def determine_initial_point(self,
                               given_params = None
                               ) -> np.ndarray:
        """
        implement TQA algorithms if None or float is given:
        https://arxiv.org/abs/2101.05742
        For float dt of TQA is calculated.
        For None Full TQA is calculated
        :param given_params: given initial parameters or None for TQA
        :return: (beta,gamma)
        """
        if given_params is None:
            self._init_params = self._tqa_beta_gamma(self._tqa_dt())
        elif isinstance(given_params, float):
            self._init_params = self._tqa_beta_gamma(given_params)
        try:
            if len(given_params) == 2 * self.p:
                self._init_params = given_params
        except:
            self._init_params = self._tqa_beta_gamma(self._tqa_dt())

    def _black_box_obj(self,
                       params: np.ndarray
                       ) -> float:
        _beta = params[:self.p]
        _gamma = params[self.p:]
        return self._qaoa_cost(_beta, _gamma)



    def _tqa_dt(self, time_1: float = 0.35, time_end: float = 1.4, time_steps_num: int = 50):
        """run TQA algorithm to find dt that determines initial gamma & beta.
        """
        dt_list = np.linspace(time_1, time_end, time_steps_num)
        cut_value = 0
        best_dt = time_1
        for dt in dt_list:
            beta, gamma = (self._tqa_beta_gamma(dt)[:self.p],
                           self._tqa_beta_gamma(dt)[self.p:])
            ## calc QAOA cost
            cost = self._qaoa_cost(beta, gamma)
            # update best_dt
            if cost <= cut_value:
                cut_value = cost
                best_dt = dt
        return best_dt

    def _tqa_beta_gamma(self,
                        best_dt: float
                        ) -> List[float]:
        """ calc concatenate list of beta, gamma from best dt
        :return: list (beta,gamma)
        """
        i_list = np.arange(1, self.p + 1) - 0.5
        gamma = (i_list / self.p) * best_dt
        beta = (1 - (i_list / self.p)) * best_dt
        return np.concatenate((beta, gamma))



    @property
    def draw_qaoa_circuit(self):
        return self._qaoa_circ.draw()

    @property
    def qaoa_circuit(self):
        return self._qaoa_circ

    @property
    def reps(self):
        return self.p

    @property
    def initial_point(self):
        try:
            return self._init_params
        except AttributeError:
            raise AttributeError('run method determine_initial_point to '
                                 'determine initial point.')















if __name__ == '__main__':
    n = 4
    a = GraphER([0,0.2,0,0.1,0,0.31], n)
    qaoaCirc = QAOAMaxCutSV(Graph=a, reps=3)#, initial_point=np.random.rand(6))
    b = GraphER([0,1,0,1,0,1], n)
    qaoaCircB = QAOAMaxCutSV(Graph=b, reps=3)

    print(a)