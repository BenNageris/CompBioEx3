import utils
from nn_genetic import GeneticNNWeightSolver

DEFAULT_MODEL = "wnet1.txt"
DEFAULT_TESTNET_FILE = "testnet1"
OUTPUT_FILE = "output-testnet1"

if __name__ == "__main__":
    # load the model trained before at buildnet1
    model = GeneticNNWeightSolver.load(path=DEFAULT_MODEL)
    # execute classification for testnet1
    utils.runnet_func(model=model, path=DEFAULT_TESTNET_FILE, output=OUTPUT_FILE)
