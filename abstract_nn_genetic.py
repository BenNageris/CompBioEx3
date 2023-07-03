from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional, Dict, Callable
import numpy as np

import activation_functions


class AbstractNNGenetic:
    """
    This class Represents the Generic Neural network solution
    """
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            hidden_sizes: List[int],
    ):
        self.nx = n_inputs
        self.ny = n_outputs
        self.hidden_sizes = hidden_sizes
        self.nh = len(hidden_sizes)
        self.sizes = [self.nx] + hidden_sizes + [self.ny]

    @staticmethod
    @abstractmethod
    def crossover(nn_1: AbstractNNGenetic, nn_2: AbstractNNGenetic):
        raise NotImplementedError()

    @abstractmethod
    def mutation(self):
        raise NotImplementedError()
