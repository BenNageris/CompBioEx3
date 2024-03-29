from dataset import DataSet
from genetic_algo import GeneticAlgo
from nn_genetic import GeneticNNWeightSolver

DUMP_TO_FILE = "wnet1.txt"

if __name__ == "__main__":
    nn_1 = DataSet.parse_nn_file(r"nn1_train.txt", "nn1_test.txt")

    genetic = GeneticAlgo(
        n_inputs=17,
        hidden_sizes=[],
        n_outputs=1,
        dataset=nn_1,
        n_population=100,
    )

    genetic.solve(n_episodes=200, should_graph=True)
    genetic._best_sol.dump(DUMP_TO_FILE)

    print(f"fitness score on test:{genetic._best_sol.fitness(nn_1.x_test, nn_1.y_test)}")

    model = GeneticNNWeightSolver.load(DUMP_TO_FILE)
    print(f"model-loaded: fitness score on test:{model.fitness(nn_1.x_test, nn_1.y_test)}")
