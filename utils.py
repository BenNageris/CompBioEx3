import typing
import numpy as np
import matplotlib.pyplot as plt
from nn_genetic import GeneticNNWeightSolver
from dataset import DataSet

Graph = typing.NamedTuple("Graph", [("graph", typing.List[float]), ("description", str), ("color", str)])


def plot_experiment(graphs: typing.List[Graph], description: str):
    """
    Plots the graph
    """
    for graph in graphs:
        size = len(graph.graph)
        plt.plot(np.arange(0, size), graph.graph, label=graph.description, color=graph.color, marker=".",
                 markersize=5)

    plt.legend()
    plt.title(description, fontsize=10)
    plt.suptitle("nn1", fontsize=20)
    plt.show()


def _get_file_content(path: str):
    """
    :param path: str
    :return: list of string, each string is a line in the file
    """
    with open(path, "r") as f:
        return f.readlines()


def runnet_func(model: GeneticNNWeightSolver, path: str, output: str) -> None:
    """
    :param model: GeneticNNWeightSolver
    :param path: str
    :param output: str
    :return: executes predictions on the bitwise string in path and output its predicted classifications into output
    """
    lines = _get_file_content(path)
    total = []
    # compute classification for each row in file
    for bitwise_str in lines:
        bitwise_str = bitwise_str.strip()
        model_input_bitwise_arr = DataSet.transform_input(bitwise_str)
        y = int(model.predict([model_input_bitwise_arr]).view())
        total.append(y)
    # write the classifications to the output file
    with open(output, "w") as f:
        for classification in total:
            f.writelines(f"{classification}\n")
