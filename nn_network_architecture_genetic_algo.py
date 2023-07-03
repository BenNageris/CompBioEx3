from typing import List, Tuple
import random
import statistics

from dataset import DataSet
from nn_architecture import GeneticNNArchitectureSolver


class NNNetworkArchitectureGeneticAlgo:
    """
    This class presents a Genetic Algorithm execution to find the optimal architecture
    """
    def __init__(
            self,
            n_population: int,
            elitism_ratio: float,
            n_network_inputs: int,
            n_network_outputs: int,
            n_episodes_per_fitness: int,
            n_generic_solver_population: int,
            dataset: DataSet,
            mutation_ratio: float = 0.3
    ):
        self._n_population = n_population
        self._elitism_ratio = elitism_ratio
        self._n_network_inputs = n_network_inputs
        self._n_network_outputs = n_network_outputs
        self._n_episodes_per_fitness = n_episodes_per_fitness
        self._n_generic_solver_population = n_generic_solver_population
        self._dataset = dataset
        self._mutation_ratio = mutation_ratio

        self._population: List[Tuple[GeneticNNArchitectureSolver, float]] = []
        for i in range(self._n_population):
            print(f"{i=}==============")
            genetic_arch_solver = GeneticNNArchitectureSolver(
                n_inputs=self._n_network_inputs,
                n_outputs=self._n_network_outputs,
                dataset=self._dataset,
                hidden_sizes=GeneticNNArchitectureSolver.randomize_hidden_size(),
            )
            fitness = genetic_arch_solver.fitness(n_episodes=self._n_episodes_per_fitness)
            print(f"fitness of {genetic_arch_solver.hidden_sizes}: {fitness}")
            self._population.append((genetic_arch_solver, fitness))

        self.sort_population()
        self.best_sol, self.best_seen_score = self._population[0]
        for genetic_arch_solver, fitness in self._population:
            print(f"{genetic_arch_solver.hidden_sizes}: {fitness}")

    def sort_population(self):
        """
        sorts the population by its fitness score
        """
        self._population = self.sort_by_fitness(self._population)

    @staticmethod
    def sort_by_fitness(generation: List[Tuple[GeneticNNArchitectureSolver, float]]):
        """
        :param generation: List of tuples, each tuple is the NN Architecture and it's fitness score
        :return: sorted generation by its fitness score
        """
        return sorted(generation, key=lambda x: x[1], reverse=True)

    def get_avg_fitness(self):
        """
        :return: average fitness of the population
        """
        return statistics.mean([fit for _, fit in self._population])

    def solve(self, n_episodes: int):
        """
        :param n_episodes: int
        """
        self.sort_population()

        elitism_ratio = 0.2
        crossover_ratio = 0.8

        for i in range(n_episodes):
            self.best_sol, self.best_seen_score = self._population[0]
            average = self.get_avg_fitness()

            print(
                (
                    f"NN generic architecture: generation {i} with best score of {self.best_seen_score},"
                    f" average fitness:{average}"
                )
            )

            next_gen = []

            # elitism
            n_elitism = max(int(elitism_ratio * self._n_population), 1)
            next_gen_sols = [self._population[i] for i in range(n_elitism)]
            next_gen.extend([next_gen_sol for next_gen_sol in next_gen_sols])

            # crossovers
            while len(next_gen) != self._n_population:
                idx = 0
                # 0, 1 or 2
                idx2 = random.choice(range(3))
                sol1, fitness_1 = self._population[idx]
                sol2, fitness_2 = self._population[idx2]
                crossover_sol = GeneticNNArchitectureSolver.crossover(sol1, sol2)
                crossover_sol_fitness = crossover_sol.fitness(n_episodes=self._n_episodes_per_fitness)
                next_gen.append((crossover_sol, crossover_sol_fitness))
                print(f"found {crossover_sol.hidden_sizes} with fitness:{crossover_sol_fitness}")
            next_gen = self.sort_by_fitness(next_gen)

            n_mutations = int(self._n_population * self._mutation_ratio)
            ratio_safe = 0.2
            safe_idx = int(ratio_safe * self._n_population)
            for mutation_count in range(n_mutations):
                idx = random.sample(range(safe_idx, self._n_population), k=1)[0]
                sol, fitness = next_gen.pop(idx)

                sol.mutation()
                # update temp best
                best_temp_sol = sol
                best_temp_fit = sol.fitness(n_episodes=self._n_episodes_per_fitness)

                next_gen.append(
                    (best_temp_sol, best_temp_fit)
                )
                print(
                    (
                        f"mutated architecture with hidden size: {best_temp_sol.hidden_sizes} "
                        f"with fitness:{best_temp_fit}"
                    )
                )
            self._population = next_gen
            self.sort_population()
            print(f"len of population:{self._population}")
