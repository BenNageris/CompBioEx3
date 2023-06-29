import utils
from nn_genetic import GeneticNNWeightSolver

DEFAULT_MODEL = "wnet0"
DEFAULT_TESTNET_FILE = "testnet0"
OUTPUT_FILE = "output-testnet0"

if __name__ == "__main__":
    model = GeneticNNWeightSolver.load(path=DEFAULT_MODEL)
    utils.runnet_func(model=model, path=DEFAULT_TESTNET_FILE, output=OUTPUT_FILE)
