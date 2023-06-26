from dataset import DataSet
from genetic_algo import GeneticAlgo

from nn_genetic import FFSN_GeneticMultiClass
from nn_architacture import NnArchitecture
DUMP_TO_FILE = "wnet0"

if __name__ == "__main__":
    nn_0 = DataSet.parse_nn_file(r"nn0.txt", p_train=0.6)

    genetic = GeneticAlgo(
        n_inputs=17,
        # hidden_sizes=[17, 8, 4],
        hidden_sizes=[8],
        # n_outputs=2,
        n_outputs=1,
        dataset=nn_0,
        n_population=100,
        tournament_winner_probability=0.5,
        tournament_size=3,
        population_type=NnArchitecture
    )

    genetic.sort_population()

    genetic.solve(n_episodes=300, should_graph=True)
    genetic._best_sol.dump(DUMP_TO_FILE)
