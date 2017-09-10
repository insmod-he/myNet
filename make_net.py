import os
from net import Net

def make_softmax_net():
    params = []
    
    layer1 = {}
    layer1["batch"]  = 128
    layer1["name"]   = "Data"
    layer1["type"]   = "MnistDataLayer"
    layer1["bottom"] = []
    layer1["top"]    = ["data", "label"]
    params.append(layer1)

    layer2 = {}
    layer2["output"] = 3      # class num
    layer2["name"] = "fc"
    layer2["type"] = "FCLayer"
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["fc1"]
    params.append(layer2)

    layer3 = {}
    layer3["class_num"] = 3
    layer3["name"]   = "softmax"
    layer3["type"]   = "SoftmaxLossLayer"
    layer3["bottom"] = ["fc1", "label"]
    layer3["top"]    = ["loss"]
    params.append(layer3)

    net = Net()
    net.init(params)
    return net

def make_2layer_mlp():
    pass

def make_2layer_cnn():
    pass