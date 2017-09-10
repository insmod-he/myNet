import os
from net import Net

def make_softmax_net():
    params = []
    
    layer1 = {}
    layer1["batch"]  = 200
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
    params = []
    
    layer1 = {}
    layer1["batch"]  = 128
    layer1["name"]   = "DATA"
    layer1["type"]   = "MnistDataLayer"
    layer1["bottom"] = []
    layer1["top"]    = ["data", "label"]
    layer1["path"]   = "./data/mnist.pkl"
    params.append(layer1)

    layer2 = {}
    layer2["output"] = 256 
    layer2["name"] = "FC1"
    layer2["type"] = "FCLayer"
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["fc1"]
    params.append(layer2)

    layer3 = {}
    layer3["name"] = "RELU1"
    layer3["type"] = "ReLULayer"
    layer3["bottom"] = ["fc1"]
    layer3["top"]    = ["relu1"]
    params.append(layer3)

    layer4 = {}
    layer4["output"] = 10      # class num
    layer4["name"] = "FC2"
    layer4["type"] = "FCLayer"
    layer4["bottom"] = ["relu1"]
    layer4["top"]    = ["fc2"]
    params.append(layer4)

    layer5 = {}
    layer5["class_num"] = 10
    layer5["name"]   = "SOFTMAXLOSS"
    layer5["type"]   = "SoftmaxLossLayer"
    layer5["bottom"] = ["fc2", "label"]
    layer5["top"]    = ["loss"]
    params.append(layer5)

    net = Net()
    net.init(params)
    return net

def make_3layer_mlp():
    params = []
    
    layer1 = {}
    layer1["batch"]  = 128
    layer1["name"]   = "DATA"
    layer1["type"]   = "MnistDataLayer"
    layer1["bottom"] = []
    layer1["top"]    = ["data", "label"]
    layer1["path"]   = "./data/mnist.pkl"
    params.append(layer1)

    layer2 = {}
    layer2["output"] = 256 
    layer2["name"] = "FC1"
    layer2["type"] = "FCLayer"
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["fc1"]
    params.append(layer2)

    layer3 = {}
    layer3["name"] = "RELU1"
    layer3["type"] = "ReLULayer"
    layer3["bottom"] = ["fc1"]
    layer3["top"]    = ["relu1"]
    params.append(layer3)

    fc2_param = {}
    fc2_param["output"] = 256 
    fc2_param["name"] = "FC2"
    fc2_param["type"] = "FCLayer"
    fc2_param["bottom"] = ["relu1"]
    fc2_param["top"]    = ["fc22"]
    params.append(fc2_param)

    relu2_param = {}
    relu2_param["name"] = "RELU2"
    relu2_param["type"] = "ReLULayer"
    relu2_param["bottom"] = ["fc22"]
    relu2_param["top"]    = ["relu2"]
    params.append(relu2_param)

    layer4 = {}
    layer4["output"] = 10      # class num
    layer4["name"] = "FC2"
    layer4["type"] = "FCLayer"
    layer4["bottom"] = ["relu2"]
    layer4["top"]    = ["fc2"]
    params.append(layer4)

    layer5 = {}
    layer5["class_num"] = 10
    layer5["name"]   = "SOFTMAXLOSS"
    layer5["type"]   = "SoftmaxLossLayer"
    layer5["bottom"] = ["fc2", "label"]
    layer5["top"]    = ["loss"]
    params.append(layer5)

    net = Net()
    net.init(params)
    return net

def make_2layer_cnn():
    pass