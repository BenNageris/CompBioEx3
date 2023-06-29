import pickle

from dataset import DataSet

if __name__ == "__main__":
    nn_1 = DataSet.parse_nn_file(None, "nn1_test.txt")

    with open(r"wnet1", "rb") as f:
        model = pickle.load(f)

    print(model.fitness(nn_1.x_test, nn_1.y_test))
