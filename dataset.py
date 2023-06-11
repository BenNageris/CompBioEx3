from __future__ import annotations

import random
from typing import  List, Optional

import numpy


class DataSet:
    P_TRAIN_DEFAULT = 0.75

    def __init__(self, nn_dataset_list: List, nn_dataset_dict):
        self._nn_dataset_list = nn_dataset_list
        self._nn_dataset_dict = nn_dataset_dict

    @staticmethod
    def parse_nn_file(path: str) -> DataSet:
        with open(path, "r") as f:
            classifications_lines = f.readlines()
        nn_dataset_dict = {}
        nn_dataset_list = []
        for row in classifications_lines:
            bitwise_txt, classification = row.split()
            nn_dataset_list.append((numpy.array([int(bit) for bit in bitwise_txt]), int(classification)))
            nn_dataset_dict[bitwise_txt] = classification
        return DataSet(
            nn_dataset_list=nn_dataset_list,
            nn_dataset_dict=nn_dataset_dict
        )

    def get_classification(self, bitwise_txt: str) -> Optional[str]:
        if bitwise_txt not in self._nn_dataset_dict:
            return None
        return self._nn_dataset_dict[bitwise_txt]

    def train_test_split(self, p_train: float):
        if p_train < 0 or p_train > 1:
            # default value
            p_train = DataSet.P_TRAIN_DEFAULT
        n_to_select = int(p_train * len(self._nn_dataset_list))
        train_indexes = set(random.sample(range(len(self._nn_dataset_list)), k=n_to_select))
        assert n_to_select == len(train_indexes), "Error, n_to_select and len of train indexes don't match"

        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for nn_dataset_idx in range(len(self._nn_dataset_list)):
            bitwise_txt, classification = self._nn_dataset_list[nn_dataset_idx]
            if nn_dataset_idx in train_indexes:
                # if was sampled
                x_train.append(bitwise_txt)
                y_train.append(classification)
            else:
                x_val.append(bitwise_txt)
                y_val.append(classification)
        return numpy.array(x_train), numpy.array(x_val), numpy.array(y_train), numpy.array(y_val)
