from activation_functions import relu, softmax, tanh, sigmoid


class DnaArchitecture:
    Depth = [3, 4, 5, 6, 7, 8, 9, 10, 1, 2]
    Layer_Size = [16, 32, 64, 128, 256, 512, 1024]
    Activations = [relu, softmax, tanh,
                   sigmoid]
    Optimizer = ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"],
    Losses = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
              "mean_squared_logarithmic_error", "squared_hinge", "hinge", "categorical_hinge", "logcosh",
              "categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy",
              "kullback_leibler_divergence", "poisson", "cosine_proximity"]
    def __init__(self,
                 depth_idx: int,
                 layer_size_idx: int,
                 activations_idx: int,
                 optimizer_idx: int,
                 losses_idx: int):
        self.depth_idx = depth_idx
        self.layer_size_idx = layer_size_idx
        self.activations_idx = activations_idx
        self.optimizer_idx = optimizer_idx
        self.losses_idx = losses_idx
