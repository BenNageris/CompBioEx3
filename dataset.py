from __future__ import annotations
from typing import  List, Optional


class DataSet:
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
            nn_dataset_list.append((str(bitwise_txt), str(classification)))
            nn_dataset_dict[bitwise_txt] = classification
        return DataSet(
            nn_dataset_list=nn_dataset_list,
            nn_dataset_dict=nn_dataset_dict
        )

    def get_classification(self, bitwise_txt: str) -> Optional[str]:
        if bitwise_txt not in self._nn_dataset_dict:
            return None
        return self._nn_dataset_dict[bitwise_txt]
