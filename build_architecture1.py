import random
from nn_network_architecture_genetic_algo import NNNetworkArchitectureGeneticAlgo
from nn_architecture import GeneticNNArchitectureSolver
from nn_genetic import GeneticNNWeightSolver
from dataset import DataSet

if __name__ == "__main__":
    nn_1 = DataSet.parse_nn_file(r"nn1_train.txt", "nn1_test.txt")

    nn_generic_architecture_solver = NNNetworkArchitectureGeneticAlgo(
        n_population=5,
        dataset=nn_1,
        n_network_inputs=17,
        n_network_outputs=1,
        n_episodes_per_fitness=50,
        n_generic_solver_population=100,
        elitism_ratio=0.1
    )
    nn_generic_architecture_solver.solve(n_episodes=3)

    best_sol = nn_generic_architecture_solver.best_sol.best_sol
    print(f"fitness of best sol:{best_sol.fitness(nn_1.x_test, nn_1.y_test)}")

    best_sol.dump("wnet1-dynamic-arch")
