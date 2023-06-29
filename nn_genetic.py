from __future__ import annotations
from typing import List, Dict, Optional, Callable

import numpy.random
from sklearn.metrics import accuracy_score
# import pickle
import json
import activation_functions
import numpy as np


class GeneticNNWeightSolver:
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_sizes: List[int],
            w: Optional[Dict[int, np.array]] = None,
            activation_func: Callable = activation_functions.binary_step
    ):
        self.nx = n_inputs
        self.ny = n_outputs
        self.hidden_sizes = hidden_sizes
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny]
        self._activation_func = activation_func
        self._w_layer_size = {}
        self._n_neurons = 0

        if w is None:
            # randomize if bias or weight isn't passed
            self.W = {}
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.uniform(-1, 1, size=(self.sizes[i], self.sizes[i + 1]))
                self._n_neurons += self.W[i + 1].size
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
        else:
            self.W = w.copy()
            for i in range(self.nh + 1):
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
                self._n_neurons += self.W[i + 1].size
        self._w_layers_idx = list(self.W.keys())
        self._max_layer_idx = max(self._w_layers_idx)

    def dump(self, path: str) -> None:
        json_dict = {
            "n_inputs": self.nx,
            "n_outputs": self.ny,
            "hidden_sizes": self.hidden_sizes,
            "w": json.dumps({k: v.tolist() for k, v in self.W.items()}),
        }
        with open(path, "wb") as f:
            json.dump(json_dict, f)

    @staticmethod
    def load(path: str) -> GeneticNNWeightSolver:
        with open(path, "rb") as f:
            json_dict = json.load(f)

        return GeneticNNWeightSolver(
            n_inputs=json_dict["n_inputs"],
            n_outputs=json_dict["n_outputs"],
            hidden_sizes=json_dict["hidden_sizes"],
            w={k: np.array(v) for k, v in json_dict["w"]},
        )

    def fitness(self, x, y):
        Y_pred = self.predict(x)

        return accuracy_score(Y_pred, y)

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1])
            self.H[i + 1] = activation_functions.binary_step(self.A[i + 1])
            # TODO: think about replacing sigmoid with step
        self.A[self.nh + 1] = np.matmul(self.H[self.nh], self.W[self.nh + 1])
        return activation_functions.binary_step(self.A[self.nh + 1])

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()

    def mutation(self):
        for layer_idx in self.W:
            layer = []
            for layer_weights in self.W[layer_idx]:
                tmp_layer = []
                for neuron in layer_weights:
                    d = numpy.random.normal() * 0.05
                    tmp_layer.append(neuron + d)
                layer.append(tmp_layer)
            self.W[layer_idx] = np.array(layer)

    @staticmethod
    def crossover(nn_1: GeneticNNWeightSolver, nn_2: GeneticNNWeightSolver) -> GeneticNNWeightSolver:
        assert nn_1._n_neurons == nn_2._n_neurons, (f"number of neurons don't match nn_1:"
                                                    f"{nn_1._n_neurons}, nn_2:{nn_2._n_neurons}")
        w = {}
        for layer_idx in nn_1.W:
            layer = []
            for layer1_weights, layer2_weights in zip(nn_1.W[layer_idx], nn_2.W[layer_idx]):
                tmp_layer = []
                for neuron_1, neuron_2 in zip(layer1_weights, layer2_weights):
                    tmp_layer.append(neuron_1 * 0.5 + neuron_2 * 0.5)
                layer.append(tmp_layer)
            w[layer_idx] = np.array(layer)

        child = GeneticNNWeightSolver(
            n_inputs=nn_1.nx,
            hidden_sizes=nn_1.hidden_sizes,
            n_outputs=nn_1.ny,
            w=w
        )
        child.mutation()
        return child
