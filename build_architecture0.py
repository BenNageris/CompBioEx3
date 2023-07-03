import random
from nn_network_architecture_genetic_algo import NNNetworkArchitectureGeneticAlgo
from nn_architecture import GeneticNNArchitectureSolver
from nn_genetic import GeneticNNWeightSolver
from dataset import DataSet

if __name__ == "__main__":
    nn_0 = DataSet.parse_nn_file(r"nn0_train.txt", "nn0_test.txt")

    nn_generic_architecture_solver = NNNetworkArchitectureGeneticAlgo(
        n_population=5,
        dataset=nn_0,
        n_network_inputs=17,
        n_network_outputs=1,
        n_episodes_per_fitness=150,
        n_generic_solver_population=100,
        elitism_ratio=0.1
    )
    nn_generic_architecture_solver.solve(n_episodes=3)

    best_sol = nn_generic_architecture_solver.best_sol.best_sol
    print(f"fitness of best sol:{best_sol.fitness(nn_0.x_test, nn_0.y_test)}")

    best_sol.dump("wnet0-dynamic-arch")
