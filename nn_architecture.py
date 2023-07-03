from __future__ import annotations

import random
from typing import List, Dict, Optional, Callable
from genetic_algo import GeneticAlgo
import numpy as np

from abstract_nn_genetic import AbstractNNGenetic
from dataset import DataSet


class GeneticNNArchitectureSolver(AbstractNNGenetic):
    """
    This class presents the Genetic Neural Network Architecture Solution Algorithm
    """
    POSSIBLE_LAYERS_SIZE = [2, 4, 6, 8]
    MAX_HIDDEN_LAYERS_SIZE = 3
    EXECUTION_EPISODES = 100
    N_POPULATION = 100

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_sizes: List[int],
            dataset: DataSet
    ):
        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
        )
        self._dataset = dataset
        self.best_sol = None

    def fitness(self, n_episodes: int = 100) -> float:
        """
        :param n_episodes: number of episodes to evaluate a NN architecture
        :return: the fitness score (float) of a NN architecture
        """
        genetic_algo = GeneticAlgo(
            n_inputs=self.nx,
            n_outputs=self.ny,
            hidden_sizes=self.hidden_sizes,
            n_population=GeneticNNArchitectureSolver.N_POPULATION,
            dataset=self._dataset,
            mutation_ratio=0.3,
        )
        genetic_algo.solve(n_episodes=n_episodes)
        self.best_sol = genetic_algo._best_sol
        return genetic_algo._best_sol.fitness(self._dataset.x_train, self._dataset.y_train)

    @staticmethod
    def crossover(nn_1: GeneticNNArchitectureSolver, nn_2: GeneticNNArchitectureSolver) -> GeneticNNArchitectureSolver:
        """
        :param nn_1: GeneticNNArchitectureSolver
        :param nn_2: GeneticNNArchitectureSolver
        :return: returns a new NN architecture offspring created using the two parents
        """
        nn_1_idx = random.choice(range(nn_1.nh)) if nn_1.nh > 0 else 0
        nn_2_idx = random.choice(range(nn_2.nh + 1))

        hidden_sizes = nn_1.hidden_sizes[:nn_1_idx] + nn_2.hidden_sizes[nn_2_idx:]

        # fix length of hidden size to max if length diverges
        while len(hidden_sizes) > GeneticNNArchitectureSolver.MAX_HIDDEN_LAYERS_SIZE:
            hidden_sizes.pop(random.randrange(len(hidden_sizes)))

        return GeneticNNArchitectureSolver(
            n_inputs=nn_1.nx,
            n_outputs=nn_2.ny,
            hidden_sizes=hidden_sizes,
            dataset=nn_1._dataset
        )

    def mutation(self):
        """
        :return: mutates the NN Architecture
        """
        if self.nh == 0:
            return
        layer_idx = random.choice(range(self.nh))
        self.hidden_sizes[layer_idx] = random.choice(GeneticNNArchitectureSolver.POSSIBLE_LAYERS_SIZE)

    @staticmethod
    def randomize_hidden_size():
        """
        :return: returns a list of a randomized NN architecture
        """
        n_layers = random.randint(0, GeneticNNArchitectureSolver.MAX_HIDDEN_LAYERS_SIZE)
        hidden_sizes = []
        while len(hidden_sizes) != n_layers:
            hidden_sizes.append(random.choice(GeneticNNArchitectureSolver.POSSIBLE_LAYERS_SIZE))
        return hidden_sizes
