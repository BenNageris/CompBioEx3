import utils
from nn_genetic import GeneticNNWeightSolver

DEFAULT_MODEL = "wnet0.txt"
DEFAULT_TESTNET_FILE = "testnet0"
OUTPUT_FILE = "output-testnet0"

if __name__ == "__main__":
    # load the model trained before at buildnet0
    model = GeneticNNWeightSolver.load(path=DEFAULT_MODEL)
    # execute classification for testnet0
    utils.runnet_func(model=model, path=DEFAULT_TESTNET_FILE, output=OUTPUT_FILE)
