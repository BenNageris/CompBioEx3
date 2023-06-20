from __future__ import annotations
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score
import tqdm
import pickle
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from abstract_genetic_nn import AbstractGeneticNN
from dataset import DataSet

class FFSN_GeneticMultiClass(AbstractGeneticNN):
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
            n_outputs=nn_1.ny,
            w=w
        )
