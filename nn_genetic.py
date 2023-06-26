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
import matplotlib.pyplot as plt

from dataset import DataSet
from nn_genetic_abstract import NNGeneticAbstract


class FFSN_GeneticMultiClass(NNGeneticAbstract):

    def mutation(self):
        # randomized_layer = random.choice(self._w_layers_idx)
        # if random.random() < 0.3:
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
    def crossover(nn_1: NNGeneticAbstract, nn_2: NNGeneticAbstract) -> NNGeneticAbstract:
        assert nn_1._n_neurons == nn_2._n_neurons, (f"number of neurons don't match nn_1:"
                                                    f"{nn_1._n_neurons}, nn_2:{nn_2._n_neurons}")
        """
        split_idx = random.randint(0, nn_1._n_neurons - 1)
        if split_idx == nn_1._n_neurons:
            # copy all from nn_1
            layer_idx, (x_loc, y_loc) = nn_1._max_layer_idx, nn_1.W[nn_1._max_layer_idx].shape
        else:
            layer_idx, x_loc, y_loc = nn_1.get_neuron_location(neuron_idx=split_idx)
        w = {}
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
        """
        # alternative
        w = {}
        for layer_idx in nn_1.W:
            layer = []
            for layer1_weights, layer2_weights in zip(nn_1.W[layer_idx], nn_2.W[layer_idx]):
                tmp_layer = []
                for neuron_1, neuron_2 in zip(layer1_weights, layer2_weights):
                    tmp_layer.append(neuron_1 * 0.5 + neuron_2 * 0.5)
                layer.append(tmp_layer)
            w[layer_idx] = np.array(layer)

        child = FFSN_GeneticMultiClass(
            n_inputs=nn_1.nx,
            hidden_sizes=nn_1.hidden_sizes,
            n_outputs=nn_1.ny,
            w=w
        )
        child.mutation()
        return child

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
