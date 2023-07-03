from __future__ import annotations
from typing import List, Dict, Tuple

import numpy


class DataSet:
    """
    The DataSet class, it hold the train and test data
    """
    def __init__(
            self,
            train_dataset_list: List,
            test_dataset_list: List
    ):
        self._train_dataset_list = train_dataset_list
        self._test_dataset_list = test_dataset_list

        self.x_train, self.y_train = self._parse_datasets(self._train_dataset_list)
        self.x_test, self.y_test = self._parse_datasets(self._test_dataset_list)

    @staticmethod
    def _parse_datasets(nn_dataset_list: List) -> Tuple[List, List]:
        x = []
        y = []
        for bitwise_str, classification in nn_dataset_list:
            x.append(bitwise_str)
            y.append(classification)
        return x, y

    @staticmethod
    def _transform_bias(bitwise_str: str) -> str:
        return bitwise_str + "1"

    @staticmethod
    def transform_input(bitwise_str: str) -> numpy.array:
        bitwise_txt = DataSet._transform_bias(bitwise_str)
        return numpy.array([int(bit) for bit in bitwise_txt])

    @staticmethod
    def _parse_input_file(path: str) -> List:
        if path is None:
            return []
        with open(path, "r") as f:
            classifications = f.readlines()
        nn_dataset_list = []
        for row in classifications:
            bitwise_txt, classification = row.split()
            nn_dataset_list.append((DataSet.transform_input(bitwise_txt), int(classification)))
        return nn_dataset_list

    @staticmethod
    def parse_nn_file(train_path: str = None, test_path: str = None) -> DataSet:
        train_dataset_list = DataSet._parse_input_file(train_path)
        test_dataset_list = DataSet._parse_input_file(test_path)

        return DataSet(
            train_dataset_list=train_dataset_list,
            test_dataset_list=test_dataset_list,
        )
