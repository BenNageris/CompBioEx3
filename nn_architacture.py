import copy
import random

from dna_architacture import DnaArchitecture
from nn_genetic_abstract import NNGeneticAbstract
class NnArchitecture(NNGeneticAbstract):
    @staticmethod
    def crossover(nn_1: NNGeneticAbstract, nn_2: NNGeneticAbstract) -> NNGeneticAbstract:
        dna1 = nn_1.dna_architecture
        dna2 = nn_2.dna_architecture
        dna_merge = DnaArchitecture(0, 0, 0, 0, 0)
        for key in dna1.__dict__.keys():
            if random.random() > 0.5:
                dna_merge.__dict__[key] = dna1.__dict__[key]
            else:
                dna_merge.__dict__[key] = dna2.__dict__[key]

        nn_merged = copy.deepcopy(nn_1)
        nn_merged.dna_architecture = dna_merge
        return nn_merged
    def mutation(self):
        for key in self.dna_architecture.__dict__.keys():
            if random.random() > 0.5:
                if random.random() > 0.5:
                    #TODO - modulo the size of the dna option list!
                    self.dna_architecture.__dict__[key] += 1
                else:
                    self.dna_architecture.__dict__[key] -= 1