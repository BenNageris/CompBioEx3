import pickle

from nn_genetic import FFSN_GeneticMultiClass
from dataset import DataSet

if __name__ == "__main__":
    nn_1 = DataSet.parse_nn_file(r"nn1.txt", p_train=0.7)

    with open(r"wnet1", "rb") as f:
        model = pickle.load(f)

    print(model.fitness(nn_1.x_val, nn_1.y_val))
