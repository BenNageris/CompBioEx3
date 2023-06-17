import typing
import numpy as np
import matplotlib.pyplot as plt
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
