from dataset import DataSet

if __name__ == "__main__":
    nn_0 = DataSet.parse_nn_file(r"nn0.txt")
    print(nn_0.get_classification("1110100000011100"))
    