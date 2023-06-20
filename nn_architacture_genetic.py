import math
from enum import Enum
from random import random

from abstract_genetic_nn import AbstractGeneticNN

class Coin(Enum):
    Head = 1
    Tail = 2

class ArchitectureGeneticMultiClass(AbstractGeneticNN):

    def mutation(self):
        mut = random.random()
        if mut < self._mutation_p:
            for layer_idx in range(1,self.nh):
                #randomly append one extra neuron
                self.W[layer_idx].append(self.W[random.choose(range[len(self.W[layer_idx])])])

    @staticmethod
    def crossover(nn_1: AbstractGeneticNN, nn_2: AbstractGeneticNN) -> AbstractGeneticNN:
        avg = math.ceil((nn_1.nh+nn_2.nh)/2)
        w = {}
        i_1 = 1
        i_2 = 1
        for i in range(1, avg):
            if i_1 > nn_1.nh and i_2 <= nn_2.nh:
                w[i] = nn_2.W[i_2]
            elif i_2 > nn_2.nh and i_1 <= nn_1.nh:
                w[i] = nn_1.W[i_1]
            else:
                w[i] = ArchitectureGeneticMultiClass.combine(nn_1, nn_2, i_1, i_2)
            i_1 += 1
            i_2 += 1

        return ArchitectureGeneticMultiClass(n_inputs=nn_1.nx,
            hidden_sizes=nn_1.hidden_sizes,
            n_outputs=nn_1.ny,
            w=w
        )
    @staticmethod
    def combine(nn_1: AbstractGeneticNN, nn_2: AbstractGeneticNN, i_1:int, i_2:int):
        coin = random.choice([Coin.Head,Coin.Tail])
        if coin == Coin.Head:
            return nn_1.W[i_1]
        return nn_2.W[i_2]
