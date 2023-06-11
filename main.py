import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from dataset import DataSet
from nn_network import FFSN_MultiClass

if __name__ == "__main__":
    enc = OneHotEncoder()

    # exercise
    nn_0 = DataSet.parse_nn_file(r"nn0.txt")
    x_train, x_val, y_train, y_val = nn_0.train_test_split(p_train=0.6)
    print(len(x_train), len(x_val), len(y_train), len(y_val))

    ffsn_multi = FFSN_MultiClass(n_inputs=16, hidden_sizes=[16, 8, 4], n_outputs=2)

    y_OH_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
    ffsn_multi.fit(x_train, y_OH_train, epochs=3000, learning_rate=.005, display_loss=True)

    Y_pred_train = ffsn_multi.predict(x_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)

    Y_pred_val = ffsn_multi.predict(x_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)

    accuracy_train = accuracy_score(Y_pred_train, y_train)
    accuracy_val = accuracy_score(Y_pred_val, y_val)

    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))
    """
    # base
    data, labels = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)
    # print(data.shape, labels.shape)
    ffsn_multi = FFSN_MultiClass(n_inputs=2, hidden_sizes=[2, 3], n_outputs=4)
    X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
    print(type(X_train))
    print(X_train.shape)
    
    # print(X_train.shape, X_val.shape)
    # 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
    y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
    # y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()

    ffsn_multi.fit(X_train, y_OH_train, epochs=100, learning_rate=.005, display_loss=True)
    """

    """
    Y_pred_train = ffsn_multi.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)

    Y_pred_val = ffsn_multi.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)

    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)

    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))

    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_pred_train, cmap=my_cmap,
                s=15 * (np.abs(np.sign(Y_pred_train - Y_train)) + .1))
    plt.show()
    """