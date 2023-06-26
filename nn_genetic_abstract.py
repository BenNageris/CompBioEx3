from __future__ import annotations
from typing import List, Dict, Optional, Callable

import numpy.random
from sklearn.metrics import accuracy_score
import tqdm
import pickle
import random
import itertools
import activation_functions
import numpy as np

from dna_architacture import DnaArchitecture


class NNGeneticAbstract:
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_sizes: List[int],
            mutation_p: float = 1.0,
            max_mutation_diff: float = 0.3,
            w: Optional[Dict[int, np.array]] = None,
            activation_func: Callable = activation_functions.binary_step,
            dna_architecture: DnaArchitecture = None
    ):
        self.nx = n_inputs
        self.ny = n_outputs
        self.hidden_sizes = hidden_sizes
        self.nh = len(hidden_sizes)
        self._max_mutation_diff = max_mutation_diff
        self.sizes = [self.nx] + hidden_sizes + [self.ny]
        self._activation_func = activation_func
        self._w_layer_size = {}
        self._n_neurons = 0
        self.dna_architecture = dna_architecture

        if w is None:
            # randomize if bias or weight isn't passed
            self.W = {}
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.uniform(-1, 1, size=(self.sizes[i], self.sizes[i + 1]))
                # self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self._n_neurons += self.W[i + 1].size
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
        else:
            # print(f"w={hex(id(w))}")
            self.W = w.copy()
            for i in range(self.nh + 1):
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
                self._n_neurons += self.W[i + 1].size
        self._w_layers_idx = list(self.W.keys())
        self._max_layer_idx = max(self._w_layers_idx)
        self._mutation_p = mutation_p

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> NNGeneticAbstract:
        with open(path, "rb") as f:
            return pickle.load(f)

    def fitness(self, x, y):
        Y_pred = self.predict(x)
        # print(Y_pred)
        # Y_pred = np.argmax(Y_pred, 1)

        return accuracy_score(Y_pred, y)

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1])
            self.H[i + 1] = self._activation_func(self.A[i + 1])
            # TODO: think about replacing sigmoid with step
            # print(f"{self.A[i + 1]}")
            # print(f"after step:{activation_functions.binary_step(self.A[i+1])}")
            # self.H[i + 1] = activation_functions.sigmoid(self.A[i + 1])
        self.A[self.nh + 1] = np.matmul(self.H[self.nh], self.W[self.nh + 1])
        # print(f"A-after-all:{self.A[self.nh+1]}")
        output = self._activation_func(self.A[self.nh + 1])
        # print(f"A-after-all:{output}")
        return output
        # self.H[self.nh + 1] = activation_functions.softmax(self.A[self.nh + 1])
        # print(f"out-after-all:{self.H[self.nh + 1]}")
        # print(self.H[self.nh + 1])
        # asdasd
        # return self.H[self.nh + 1]

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()

    def mutation(self):
        raise NotImplementedError

    @staticmethod
    def crossover(nn_1: NNGeneticAbstract, nn_2: NNGeneticAbstract) -> NNGeneticAbstract:
        raise NotImplementedError

    def get_neuron_location(self, neuron_idx: int):
        layers = sorted(self.W.keys())
        n_neurons_passed = 0
        for layer in layers:
            n_current_layer = self.W[layer].size
            if neuron_idx < n_neurons_passed + n_current_layer:
                idx = neuron_idx - n_neurons_passed
                layer_x_size, layer_y_size = self.W[layer].shape
                return layer, int(idx / layer_y_size), idx % layer_y_size
            n_neurons_passed += n_current_layer
        return None

    @staticmethod
    def cross_entropy(label, pred):
        yl = np.multiply(pred, label)
        yl = yl[yl != 0]
        yl = -np.log(yl)
        yl = np.mean(yl)
        return yl

    def regenerate_network(self):
        self._activation_func = DnaArchitecture.Activations[self.dna_architecture.activations_idx \
            % len(DnaArchitecture.Activations)]
        self.nh = DnaArchitecture.Depth[self.dna_architecture.depth_idx \
            % len(DnaArchitecture.Depth)]
        new_size = DnaArchitecture.Layer_Size[self.dna_architecture.layer_size_idx \
            % len(DnaArchitecture.Layer_Size)]
        for size in self.sizes:
            if random.random() < 0.2:
                size = new_size

        self.W = {}
        if self.W is None:
            # randomize if bias or weight isn't passed
            self.W = {}
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.uniform(-1, 1, size=(self.sizes[i], self.sizes[i + 1]))
                self._n_neurons += self.W[i + 1].size
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
        else:
            self.W = self.W.copy()
            for i in range(self.nh + 1):
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
                self._n_neurons += self.W[i + 1].size

