import cPickle as PKL

def load_mnist(path="./data/mnist.pkl"):
    fd = open(path, "r")
    R  = PKL.load(fd)
    fd.close()

    mnist = {}
    mnist["train_data"] = R[0][0]
    mnist["train_lab"]  = R[0][1]
    mnist["val_data"]   = R[1][0]
    mnist["val_lab"]    = R[1][1]
    mnist["test_data"]  = R[2][0]
    mnist["test_lab"]   = R[2][1]

    return mnist