import typing
import random
import statistics
from typing import List, Tuple
from nn_genetic import FFSN_GeneticMultiClass
from dataset import DataSet
from utils import Graph, plot_experiment


class GeneticAlgo:
    def __init__(
            self,
            n_population,
            dataset: DataSet,
            n_inputs=17,
            # hidden_sizes=[16, 8, 4],
            hidden_sizes=[],
            n_outputs=2,
            tournament_winner_probability: float = 0.5,
            tournament_size: int = 5,
    ):
        self._n_population = n_population
        self._dataset = dataset
        self._population: List[Tuple[FFSN_GeneticMultiClass, float]] = []
        self._n_inputs = n_inputs
        self._hidden_sizes = hidden_sizes
        self._n_outputs = n_outputs
        self._tournament_winner_probability = tournament_winner_probability
        self._tournament_size = tournament_size

        self._tournament_prob = [self._tournament_winner_probability]
        for i in range(1, self._tournament_size):
            self._tournament_prob.append(self._tournament_prob[i - 1] * (1.0 - self._tournament_winner_probability))

        for i in range(self._n_population):
            sol = FFSN_GeneticMultiClass(
                    n_inputs=self._n_inputs,
                    hidden_sizes=self._hidden_sizes,
                    n_outputs=self._n_outputs,
                )
            self._population.append(
                (sol, sol.fitness(self._dataset.x_train, self._dataset.y_train))
            )
        self.sort_population()
        self._best_sol , self._best_seen_score = self._population[0]
        print(f"best seen solution:{self._best_sol}, best seen score:{self._best_seen_score}")
        print(self._population)

    def sort_population(self):
        self._population = self.sort_by_fitness(self._population)

    @staticmethod
    def sort_by_fitness(generation: List[Tuple[FFSN_GeneticMultiClass, float]]):
        return sorted(generation, key=lambda x: x[1], reverse=True)

    def get_avg_fitness(self):
        return statistics.mean([fit for _, fit in self._population])

    def solve(self, n_episodes: int, should_graph: bool = False):
        self.sort_population()

        best_sol_fitness_stat = []
        avg_sol_fitness_stat = []

        elitism_ratio = 0.1
        crossover_ratio = 0.9
        mutation_ratio = 0.2
        for i in range(n_episodes):
            self._best_sol, self._best_seen_score = self._population[0]

            average = self.get_avg_fitness()
            if should_graph:
                best_sol_fitness_stat.append(self._best_seen_score)
                avg_sol_fitness_stat.append(average)

            next_gen = []
            print(f"generation {i} with best score of {self._best_seen_score}, average fitness:{average}")

            # elitism
            n_elitism = int(elitism_ratio * self._n_population)
            next_gen_sols = [self._population[i] for i in range(n_elitism)]
            next_gen.extend([next_gen_sol for next_gen_sol in next_gen_sols])

            # crossovers
            crossover_idx = 0
            while len(next_gen) != self._n_population:
                indexes = sorted(random.sample(range(self._n_population), k=self._tournament_size))
                idx = random.choices(indexes, k=1, weights=self._tournament_prob)[0]
                indexes_2 = sorted(random.sample(range(self._n_population), k=self._tournament_size))
                idx2 = random.choices(indexes_2, k=1, weights=self._tournament_prob)[0]
                sol1, fitness_1 = self._population[idx]
                sol2, fitness_2 = self._population[idx2]
                # TODO:: solve the bias problem!!!!! crossover doesn't set bias
                crossover_sol = FFSN_GeneticMultiClass.crossover(sol1, sol2)
                crossover_sol_fitness = crossover_sol.fitness(self._dataset.x_train, self._dataset.y_train)
                next_gen.append((crossover_sol, crossover_sol_fitness))
                crossover_idx += 1

            next_gen = self.sort_by_fitness(next_gen)

            n_mutations = int(self._n_population * mutation_ratio)
            ratio_safe = 0.1
            safe_idx = int(ratio_safe * self._n_population)
            for mutation_count in range(n_mutations):
                idx = random.sample(range(safe_idx, self._n_population), k=1)[0]
                sol, fitness = next_gen.pop(idx)
                sol.mutation()
                # print(f"len of pop before:{len(self._population)}")
                next_gen.append(
                    (sol, sol.fitness(self._dataset.x_train, self._dataset.y_train))
                )
                # print(f"len of pop:{len(self._population)}")
            self._population = next_gen
            print(f"len of population:{self._population}")
            self.sort_population()

        if should_graph:
            plot_experiment(
                graphs=[
                    Graph(best_sol_fitness_stat, "Best solution prediction rate", "red"),
                    Graph(avg_sol_fitness_stat, "Average Prediction rate", "blue")
                ],
                description="Genetic algorithm summarized scores"
            )