import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from dataset import DataSet
from nn_network import FFSN_MultiClass
from nn_genetic import FFSN_GeneticMultiClass

if __name__ == "__main__":
    nn_0 = DataSet.parse_nn_file(r"nn0.txt")
    x_train, x_val, y_train, y_val = nn_0.train_test_split(p_train=0.6)

    genetic_nn_1 = FFSN_GeneticMultiClass(
        n_inputs=16,
        # hidden_sizes=[4, 2],
        hidden_sizes=[16, 8, 4],
        n_outputs=2,
    )
    print(genetic_nn_1.fitness(x_train, y_train))

    print(f"nn1:{genetic_nn_1.W}")
    genetic_nn_2 = FFSN_GeneticMultiClass(
        n_inputs=16,
        # hidden_sizes=[4, 2],
        hidden_sizes=[16, 8, 4],
        n_outputs=2,
    )
    print(f"nn2:{genetic_nn_2.W}")

    genetic_crossover_nn_3 = FFSN_GeneticMultiClass.crossover(genetic_nn_1, genetic_nn_2)
    print(f"nn3:{genetic_crossover_nn_3.W}")



    # print(genetic_nn._n_neurons)
    # layer, x, y = genetic_nn.get_neuron_location(423)

    # genetic_nn.mutation()

    # print(genetic_algo.W)
    """
    enc = OneHotEncoder()

    # exercise
    nn_0 = DataSet.parse_nn_file(r"nn1.txt")
    # nn_0 = DataSet.parse_nn_file(r"nn1.txt")
    x_train, x_val, y_train, y_val = nn_0.train_test_split(p_train=0.6)
    print(len(x_train), len(x_val), len(y_train), len(y_val))

    ffsn_multi = FFSN_MultiClass(n_inputs=16, hidden_sizes=[16, 8, 4], n_outputs=2)

    y_OH_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
    ffsn_multi.fit(x_train, y_OH_train, epochs=1000, learning_rate=.005, display_loss=True)

    Y_pred_train = ffsn_multi.predict(x_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)

    Y_pred_val = ffsn_multi.predict(x_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)

    accuracy_train = accuracy_score(Y_pred_train, y_train)
    accuracy_val = accuracy_score(Y_pred_val, y_val)

    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))
    """