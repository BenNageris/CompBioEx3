from dataset import DataSet
from genetic_algo import GeneticAlgo

DUMP_TO_FILE = "wnet1"

if __name__ == "__main__":
    nn_1 = DataSet.parse_nn_file(r"nn1.txt", p_train=0.7)

    genetic = GeneticAlgo(
        n_inputs=17,
        hidden_sizes=[],
        n_outputs=2,
        dataset=nn_1,
        n_population=50,
        tournament_winner_probability=0.5,
        tournament_size=5
    )
    genetic.solve(n_episodes=150, should_graph=True)
    genetic._best_sol.dump(DUMP_TO_FILE)

