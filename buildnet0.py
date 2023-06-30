from dataset import DataSet
from genetic_algo import GeneticAlgo
from nn_genetic import GeneticNNWeightSolver

DUMP_TO_FILE = "wnet0"

if __name__ == "__main__":
    nn_0 = DataSet.parse_nn_file(r"nn0_train.txt", "nn0_test.txt")

    genetic = GeneticAlgo(
        n_inputs=17,
        hidden_sizes=[8],
        n_outputs=1,
        dataset=nn_0,
        n_population=100,
    )

    genetic.solve(n_episodes=300, should_graph=True)
    genetic._best_sol.dump(DUMP_TO_FILE)

    print(f"fitness score on test:{genetic._best_sol.fitness(nn_0.x_test, nn_0.y_test)}")

    model = GeneticNNWeightSolver.load(DUMP_TO_FILE)
    print(f"model-loaded: fitness score on test:{model.fitness(nn_0.x_test, nn_0.y_test)}")
