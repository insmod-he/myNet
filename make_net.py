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
    #layer1["path"]   = "./data/mnist.pkl"
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
    params = []
    
    layer1 = {}
    layer1["batch"]  = 8
    layer1["name"]   = "DATA"
    layer1["type"]   = "MnistDataLayer"
    layer1["bottom"] = []
    layer1["top"]    = ["data", "label"]
    layer1["path"]   = "./data/mnist.pkl"
    params.append(layer1)

    layer2 = {}
    layer2["output"] = 16 
    layer2["name"] = "CONV1"
    layer2["type"] = "ConvLayer"
    layer2["kernel_size"] = 3
    layer2["pad"]  = 1
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["conv1"]
    params.append(layer2)

    layer3 = {}
    layer3["name"] = "RELU1"
    layer3["type"] = "ReLULayer"
    layer3["bottom"] = ["conv1"]
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

def make_pooling_cnn():
    params = []
    
    layer1 = {}
    layer1["batch"]  = 32
    layer1["name"]   = "DATA"
    layer1["type"]   = "MnistDataLayer"
    layer1["bottom"] = []
    layer1["top"]    = ["data", "label"]
    layer1["path"]   = "./data/mnist.pkl"
    params.append(layer1)

    layer2 = {}
    layer2["output"] = 128 
    layer2["name"] = "CONV1"
    layer2["type"] = "ConvLayer"
    layer2["kernel_size"] = 3
    layer2["pad"]  = 1
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["conv1"]
    params.append(layer2)

    layer3 = {}
    layer3["name"] = "RELU1"
    layer3["type"] = "ReLULayer"
    layer3["bottom"] = ["conv1"]
    layer3["top"]    = ["relu1"]
    params.append(layer3)

    layer4 = {}
    layer4["name"] = "POOLING1"
    layer4["type"] = "PoolingLayer"
    layer4["bottom"] = ["relu1"]
    layer4["top"]    = ["pooling1"]
    params.append(layer4)

    layer5 = {}
    layer5["output"] = 10      # class num
    layer5["name"] = "FC2"
    layer5["type"] = "FCLayer"
    layer5["bottom"] = ["pooling1"]
    layer5["top"]    = ["fc2"]
    params.append(layer5)

    layer6 = {}
    layer6["class_num"] = 10
    layer6["name"]   = "SOFTMAXLOSS"
    layer6["type"]   = "SoftmaxLossLayer"
    layer6["bottom"] = ["fc2", "label"]
    layer6["top"]    = ["loss"]
    params.append(layer6)

    net = Net()
    net.init(params)
    return net


def make_LeNet():
    params = []
    
    layer1 = {}
    layer1["batch"]  = 64
    layer1["name"]   = "DATA"
    layer1["type"]   = "MnistDataLayer"
    layer1["bottom"] = []
    layer1["top"]    = ["data", "label"]
    layer1["path"]   = "./data/mnist.pkl"
    params.append(layer1)

    layer2 = {}
    layer2["output"] = 20 
    layer2["name"] = "CONV1"
    layer2["type"] = "ConvLayer"
    layer2["kernel_size"] = 5
    layer2["pad"]  = 0
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["conv1"]
    params.append(layer2)

    layer3 = {}
    layer3["name"] = "RELU1"
    layer3["type"] = "ReLULayer"
    layer3["bottom"] = ["conv1"]
    layer3["top"]    = ["relu1"]
    params.append(layer3)

    layer4 = {}
    layer4["name"] = "POOL1"
    layer4["type"] = "PoolingLayer"
    layer4["bottom"] = ["relu1"]
    layer4["top"]    = ["pooling1"]
    params.append(layer4)

    # conv2
    layer_conv2 = {}
    layer_conv2["name"] = "CONV2"
    layer_conv2["type"] = "ConvLayer"
    layer_conv2["output"] = 50
    layer_conv2["kernel_size"] = 5
    layer_conv2["pad"]    = 0
    layer_conv2["bottom"] = ["pooling1"]
    layer_conv2["top"]    = ["conv2"]
    params.append(layer_conv2)

    # relu_conv2
    relu2_param = {}
    relu2_param["name"] = "RELU2"
    relu2_param["type"] = "ReLULayer"
    relu2_param["bottom"] = ["conv2"]
    relu2_param["top"]    = ["relu2"]
    params.append(relu2_param)

    # pool2
    pool2_param = {}
    pool2_param["name"] = "POOL2"
    pool2_param["type"] = "PoolingLayer"
    pool2_param["bottom"] = ["relu2"]
    pool2_param["top"]    = ["pool2"]
    params.append(pool2_param)

    # fc1
    fc1_param = {}
    fc1_param["name"] = "FC1"
    fc1_param["type"] = "FCLayer"
    fc1_param["output"] = 500
    fc1_param["bottom"] = ["pool2"]
    fc1_param["top"]    = ["fc1"]
    params.append(fc1_param)

    # fc1_relu
    fc1_relu_param = {}
    fc1_relu_param["name"] = "FC1_RELU"
    fc1_relu_param["type"] = "ReLULayer"
    fc1_relu_param["bottom"] = ["fc1"]
    fc1_relu_param["top"]    = ["fc1_relu"]
    params.append(fc1_relu_param)

    layer5 = {}
    layer5["output"] = 10      # class num
    layer5["name"] = "FC2"
    layer5["type"] = "FCLayer"
    layer5["bottom"] = ["fc1_relu"]
    layer5["top"]    = ["fc2"]
    params.append(layer5)

    layer6 = {}
    layer6["class_num"] = 10
    layer6["name"]   = "SOFTMAXLOSS"
    layer6["type"]   = "SoftmaxLossLayer"
    layer6["bottom"] = ["fc2", "label"]
    layer6["top"]    = ["loss"]
    params.append(layer6)

    net = Net()
    net.init(params)
    return net