from __future__ import annotations
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score
import tqdm
import pickle
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from dataset import DataSet


class FFSN_GeneticMultiClass:
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_sizes: List[int],
            mutation_p: float = 0.05,
            max_mutation_diff: float = 0.2,
            w: Optional[Dict[int, np.array]] = None,
            b: Optional[Dict[int, np.array]] = None):
        self.nx = n_inputs
        self.ny = n_outputs
        self.hidden_sizes = hidden_sizes
        self.nh = len(hidden_sizes)
        self._max_mutation_diff = max_mutation_diff
        self.sizes = [self.nx] + hidden_sizes + [self.ny]
        self._w_layer_size = {}

        if w is None or b is None:
            # randomize if bias or weight isn't passed
            self.W = {}
            self.B = {}
            self._n_neurons = 0
            for i in range(self.nh + 1):
                self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self._n_neurons += self.W[i + 1].size
                # TODO:: consider change to np.ones
                self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
        else:
            self.W = w
            self.B = b
            for i in range(self.nh + 1):
                self._w_layer_size[i + 1] = self.sizes[i] * self.sizes[i + 1]
        self._w_layers_idx = list(self.W.keys())
        self._max_layer_idx = max(self._w_layers_idx)
        self._mutation_p = mutation_p

    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> FFSN_GeneticMultiClass:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def softmax(x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def fitness(self, x, y):
        Y_pred = self.predict(x)
        Y_pred = np.argmax(Y_pred, 1)

        return accuracy_score(Y_pred, y)

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        for i in range(self.nh):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
            self.H[i + 1] = self.sigmoid(self.A[i + 1])
        self.A[self.nh + 1] = np.matmul(self.H[self.nh], self.W[self.nh + 1]) + self.B[self.nh + 1]
        self.H[self.nh + 1] = self.softmax(self.A[self.nh + 1])
        return self.H[self.nh + 1]

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()

    @staticmethod
    def grad_sigmoid(x):
        return x * (1 - x)

    def mutation(self):
        mut = random.random()
        if mut < self._mutation_p:
            randomized_layer = random.choice(self._w_layers_idx)
            layer = []
            for layer_weights in self.W[randomized_layer]:
                tmp_layer = []
                for neuron in layer_weights:
                    d = random.uniform(-1 * self._max_mutation_diff, self._max_mutation_diff)
                    tmp_layer.append(neuron + d)
                layer.append(tmp_layer)
            self.W[randomized_layer] = np.array(layer)

    @staticmethod
    def crossover(nn_1: FFSN_GeneticMultiClass, nn_2: FFSN_GeneticMultiClass) -> FFSN_GeneticMultiClass:
        assert nn_1._n_neurons == nn_2._n_neurons, (f"number of neurons don't match nn_1:"
                                                    f"{nn_1._n_neurons}, nn_2:{nn_2._n_neurons}")
        split_idx = random.randint(0, nn_1._n_neurons - 1)
        if split_idx == nn_1._n_neurons:
            # copy all from nn_1
            layer_idx, (x_loc, y_loc) = nn_1._max_layer_idx, nn_1.W[nn_1._max_layer_idx].shape
        else:
            layer_idx, x_loc, y_loc = nn_1.get_neuron_location(neuron_idx=split_idx)
        w = {}
        b = {}
        # copy until the split layer - nn1
        for layer in range(1, layer_idx):
            w[layer] = nn_1.W[layer]

        # split layer copy - nn1 & nn2
        split_layer_x_size, split_layer_y_size = nn_1.W[layer_idx].shape
        found = False
        layer_weights = []
        for x in range(split_layer_x_size):
            tmp_layer_weights = []
            for y in range(split_layer_y_size):
                if x == x_loc and y == y_loc:
                    found = True
                nn_to_query = nn_2 if found else nn_1
                tmp_layer_weights.append(nn_to_query.W[layer_idx][x][y])
            layer_weights.append(tmp_layer_weights)
        w[layer_idx] = np.array(layer_weights)

        # post split layers - nn2
        for layer in range(layer_idx + 1, nn_2._max_layer_idx + 1):
            w[layer] = nn_2.W[layer]

        return FFSN_GeneticMultiClass(
            n_inputs=nn_1.nx,
            hidden_sizes=nn_1.hidden_sizes,
            n_outputs=nn_2.ny,
            w=w,
            b={}
        )

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

