import typing
import numpy as np
import matplotlib.pyplot as plt
from nn_genetic import GeneticNNWeightSolver
from dataset import DataSet

Graph = typing.NamedTuple("Graph", [("graph", typing.List[float]), ("description", str), ("color", str)])


def plot_experiment(graphs: typing.List[Graph], description: str):
    for graph in graphs:
        size = len(graph.graph)
        plt.plot(np.arange(0, size), graph.graph, label=graph.description, color=graph.color, marker=".",
                 markersize=5)

    plt.legend()
    plt.title(description, fontsize=10)
    plt.suptitle("nn1", fontsize=20)
    plt.show()


def _get_file_content(path: str):
    with open(path, "r") as f:
        return f.readlines()


def runnet_func(model: GeneticNNWeightSolver, path: str, output: str) -> None:
    lines = _get_file_content(path)
    total = []
    for bitwise_str in lines:
        bitwise_str = bitwise_str.strip()
        model_input_bitwise_arr = DataSet.transform_input(bitwise_str)
        y = int(model.predict([model_input_bitwise_arr]).view())
        total.append((bitwise_str, y))

    with open(output, "w") as f:
        for bitwise_str, classification in total:
            f.writelines(f"{bitwise_str} {classification}\n")
