import numpy as np


def sigmoid(x):
    # print(f"sigmoid:{x}")
    return 1.0 / (1.0 + np.exp(-x))


# binary_step = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float64])

def binary_step(x):
    # print(f"step:{x}")
    step_func = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float64])
    return step_func(x)


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)
