from __future__ import annotations
from typing import List, Dict, Tuple

import numpy


class DataSet:
    def __init__(
            self,
            train_dataset_list: List,
            train_dataset_dict: Dict,
            test_dataset_list: List,
            test_dataset_dict: Dict
    ):
        self._train_dataset_list = train_dataset_list
        self._train_dataset_dict = train_dataset_dict
        self._test_dataset_list = test_dataset_list
        self._test_dataset_dict = test_dataset_dict

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
    def _parse_input_file(path: str) -> Tuple[List, Dict]:
        if path is None:
            return [], {}
        with open(path, "r") as f:
            classifications = f.readlines()
        nn_dataset_dict = {}
        nn_dataset_list = []
        for row in classifications:
            bitwise_txt, classification = row.split()
            bitwise_txt = DataSet._transform_bias(bitwise_txt)
            nn_dataset_list.append((numpy.array([int(bit) for bit in bitwise_txt]), int(classification)))
            nn_dataset_dict[bitwise_txt] = classification
        return nn_dataset_list, nn_dataset_dict

    @staticmethod
    def parse_nn_file(train_path: str = None, test_path: str = None) -> DataSet:
        train_dataset_list, train_dataset_dict = DataSet._parse_input_file(train_path)
        test_dataset_list, test_dataset_dict = DataSet._parse_input_file(test_path)

        return DataSet(
            train_dataset_list=train_dataset_list,
            train_dataset_dict=train_dataset_dict,
            test_dataset_list=test_dataset_list,
            test_dataset_dict=test_dataset_dict
        )
