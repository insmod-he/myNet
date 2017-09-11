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

def make_LeNet9():
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
    layer2["output"] = 24 
    layer2["name"] = "CONV1"
    layer2["type"] = "ConvLayer"
    layer2["kernel_size"] = 5
    layer2["pad"]  = 2
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
    layer_conv2["output"] = 48
    layer_conv2["kernel_size"] = 3
    layer_conv2["pad"]    = 1
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

    # conv3
    layer_conv3 = {}
    layer_conv3["name"] = "CONV3"
    layer_conv3["type"] = "ConvLayer"
    layer_conv3["output"] = 48
    layer_conv3["kernel_size"] = 3
    layer_conv3["pad"]    = 1
    layer_conv3["bottom"] = ["relu2"]
    layer_conv3["top"]    = ["conv3"]
    params.append(layer_conv3)

    # relu_conv3
    relu3_param = {}
    relu3_param["name"] = "RELU3"
    relu3_param["type"] = "ReLULayer"
    relu3_param["bottom"] = ["conv3"]
    relu3_param["top"]    = ["relu3"]
    params.append(relu3_param)

    # conv4
    layer_conv4 = {}
    layer_conv4["name"] = "CONV4"
    layer_conv4["type"] = "ConvLayer"
    layer_conv4["output"] = 48
    layer_conv4["kernel_size"] = 3
    layer_conv4["pad"]    = 1
    layer_conv4["bottom"] = ["relu3"]
    layer_conv4["top"]    = ["conv4"]
    params.append(layer_conv4)

    # relu_conv4
    relu4_param = {}
    relu4_param["name"] = "RELU4"
    relu4_param["type"] = "ReLULayer"
    relu4_param["bottom"] = ["conv4"]
    relu4_param["top"]    = ["relu4"]
    params.append(relu4_param)

    # pool2
    pool2_param = {}
    pool2_param["name"] = "POOL2"
    pool2_param["type"] = "PoolingLayer"
    pool2_param["bottom"] = ["relu4"]
    pool2_param["top"]    = ["pool2"]
    params.append(pool2_param)

    # conv5
    layer_conv5 = {}
    layer_conv5["name"] = "CONV5"
    layer_conv5["type"] = "ConvLayer"
    layer_conv5["output"] = 96
    layer_conv5["kernel_size"] = 3
    layer_conv5["pad"]    = 1
    layer_conv5["bottom"] = ["pool2"]
    layer_conv5["top"]    = ["conv5"]
    params.append(layer_conv5)

    # relu_conv5
    relu5_param = {}
    relu5_param["name"] = "RELU5"
    relu5_param["type"] = "ReLULayer"
    relu5_param["bottom"] = ["conv5"]
    relu5_param["top"]    = ["relu5"]
    params.append(relu5_param)

    # conv6
    layer_conv6 = {}
    layer_conv6["name"] = "CONV6"
    layer_conv6["type"] = "ConvLayer"
    layer_conv6["output"] = 96
    layer_conv6["kernel_size"] = 3
    layer_conv6["pad"]    = 1
    layer_conv6["bottom"] = ["relu5"]
    layer_conv6["top"]    = ["conv6"]
    params.append(layer_conv6)

    # relu_conv6
    relu6_param = {}
    relu6_param["name"] = "RELU6"
    relu6_param["type"] = "ReLULayer"
    relu6_param["bottom"] = ["conv6"]
    relu6_param["top"]    = ["relu6"]
    params.append(relu6_param)

    # conv7
    layer_conv7 = {}
    layer_conv7["name"] = "CONV7"
    layer_conv7["type"] = "ConvLayer"
    layer_conv7["output"] = 96
    layer_conv7["kernel_size"] = 3
    layer_conv7["pad"]    = 1
    layer_conv7["bottom"] = ["relu6"]
    layer_conv7["top"]    = ["conv7"]
    params.append(layer_conv7)

    # relu_conv7
    relu7_param = {}
    relu7_param["name"] = "RELU7"
    relu7_param["type"] = "ReLULayer"
    relu7_param["bottom"] = ["conv7"]
    relu7_param["top"]    = ["relu7"]
    params.append(relu7_param)


    layer5 = {}
    layer5["output"] = 10      # class num
    layer5["name"] = "FC"
    layer5["type"] = "FCLayer"
    layer5["bottom"] = ["relu7"]
    layer5["top"]    = ["fc"]
    params.append(layer5)

    layer6 = {}
    layer6["class_num"] = 10
    layer6["name"]   = "SOFTMAXLOSS"
    layer6["type"]   = "SoftmaxLossLayer"
    layer6["bottom"] = ["fc", "label"]
    layer6["top"]    = ["loss"]
    params.append(layer6)

    net = Net()
    net.init(params)
    return net

def make_LeNet7():
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
    layer2["output"] = 24 
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
    layer_conv2["output"] = 48
    layer_conv2["kernel_size"] = 3
    layer_conv2["pad"]    = 1
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

    # conv3
    layer_conv3 = {}
    layer_conv3["name"] = "CONV3"
    layer_conv3["type"] = "ConvLayer"
    layer_conv3["output"] = 48
    layer_conv3["kernel_size"] = 3
    layer_conv3["pad"]    = 1
    layer_conv3["bottom"] = ["relu2"]
    layer_conv3["top"]    = ["conv3"]
    params.append(layer_conv3)

    # relu_conv3
    relu3_param = {}
    relu3_param["name"] = "RELU3"
    relu3_param["type"] = "ReLULayer"
    relu3_param["bottom"] = ["conv3"]
    relu3_param["top"]    = ["relu3"]
    params.append(relu3_param)

    # conv4
    layer_conv4 = {}
    layer_conv4["name"] = "CONV4"
    layer_conv4["type"] = "ConvLayer"
    layer_conv4["output"] = 48
    layer_conv4["kernel_size"] = 3
    layer_conv4["pad"]    = 1
    layer_conv4["bottom"] = ["relu3"]
    layer_conv4["top"]    = ["conv4"]
    params.append(layer_conv4)

    # relu_conv4
    relu4_param = {}
    relu4_param["name"] = "RELU4"
    relu4_param["type"] = "ReLULayer"
    relu4_param["bottom"] = ["conv4"]
    relu4_param["top"]    = ["relu4"]
    params.append(relu4_param)

    # pool2
    pool2_param = {}
    pool2_param["name"] = "POOL2"
    pool2_param["type"] = "PoolingLayer"
    pool2_param["bottom"] = ["relu4"]
    pool2_param["top"]    = ["pool2"]
    params.append(pool2_param)

    # dropout1
    drop1_param = {}
    drop1_param["name"] = "DROP1"
    drop1_param["type"] = "DropoutLayer"
    drop1_param["bottom"] = ["pool2"]
    drop1_param["top"]    = ["drop1"]
    drop1_param["keep_rate"] = 0.5
    params.append(drop1_param)

    # fc1
    fc1_param = {}
    fc1_param["name"] = "FC1"
    fc1_param["type"] = "FCLayer"
    fc1_param["output"] = 256
    fc1_param["bottom"] = ["drop1"]
    fc1_param["top"]    = ["fc1"]
    params.append(fc1_param)

    # fc1_relu
    fc1_relu_param = {}
    fc1_relu_param["name"] = "FC1_RELU"
    fc1_relu_param["type"] = "ReLULayer"
    fc1_relu_param["bottom"] = ["fc1"]
    fc1_relu_param["top"]    = ["fc1_relu"]
    params.append(fc1_relu_param)

    # dropout2
    drop2_param = {}
    drop2_param["name"] = "DROP2"
    drop2_param["type"] = "DropoutLayer"
    drop2_param["bottom"] = ["fc1_relu"]
    drop2_param["top"]    = ["drop2"]
    drop2_param["keep_rate"] = 0.5
    params.append(drop2_param)

    layer5 = {}
    layer5["output"] = 10      # class num
    layer5["name"] = "FC2"
    layer5["type"] = "FCLayer"
    layer5["bottom"] = ["drop2"]
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


def make_LeNet5():
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
    layer2["output"] = 32 
    layer2["name"] = "CONV1"
    layer2["type"] = "ConvLayer"
    layer2["kernel_size"] = 5
    layer2["pad"]  = 2
    layer2["bottom"] = ["data"]
    layer2["top"]    = ["conv1"]
    params.append(layer2)

    layer3 = {}
    layer3["name"] = "RELU1"
    layer3["type"] = "ReLULayer"
    layer3["bottom"] = ["conv1"]
    layer3["top"]    = ["relu1"]
    params.append(layer3)

    # conv2
    layer_conv2 = {}
    layer_conv2["name"] = "CONV2"
    layer_conv2["type"] = "ConvLayer"
    layer_conv2["output"] = 32
    layer_conv2["kernel_size"] = 5
    layer_conv2["pad"]    = 2
    layer_conv2["bottom"] = ["relu1"]
    layer_conv2["top"]    = ["conv2"]
    params.append(layer_conv2)

    # relu_conv2
    relu2_param = {}
    relu2_param["name"] = "RELU2"
    relu2_param["type"] = "ReLULayer"
    relu2_param["bottom"] = ["conv2"]
    relu2_param["top"]    = ["relu2"]
    params.append(relu2_param)

    layer4 = {}
    layer4["name"] = "POOL1"
    layer4["type"] = "PoolingLayer"
    layer4["bottom"] = ["relu2"]
    layer4["top"]    = ["pooling1"]
    params.append(layer4)

    # conv3
    layer_conv3 = {}
    layer_conv3["name"] = "CONV3"
    layer_conv3["type"] = "ConvLayer"
    layer_conv3["output"] = 64
    layer_conv3["kernel_size"] = 3
    layer_conv3["pad"]    = 1
    layer_conv3["bottom"] = ["pooling1"]
    layer_conv3["top"]    = ["conv3"]
    params.append(layer_conv3)

    # relu_conv3
    relu3_param = {}
    relu3_param["name"] = "RELU3"
    relu3_param["type"] = "ReLULayer"
    relu3_param["bottom"] = ["conv3"]
    relu3_param["top"]    = ["relu3"]
    params.append(relu3_param)

    # conv4
    layer_conv4 = {}
    layer_conv4["name"] = "CONV4"
    layer_conv4["type"] = "ConvLayer"
    layer_conv4["output"] = 64
    layer_conv4["kernel_size"] = 3
    layer_conv4["pad"]    = 1
    layer_conv4["bottom"] = ["relu3"]
    layer_conv4["top"]    = ["conv4"]
    params.append(layer_conv4)

    # relu_conv4
    relu4_param = {}
    relu4_param["name"] = "RELU4"
    relu4_param["type"] = "ReLULayer"
    relu4_param["bottom"] = ["conv4"]
    relu4_param["top"]    = ["relu4"]
    params.append(relu4_param)

    # pool2
    pool2_param = {}
    pool2_param["name"] = "POOL2"
    pool2_param["type"] = "PoolingLayer"
    pool2_param["bottom"] = ["relu4"]
    pool2_param["top"]    = ["pool2"]
    params.append(pool2_param)

    # dropout1
    drop1_param = {}
    drop1_param["name"] = "DROP1"
    drop1_param["type"] = "DropoutLayer"
    drop1_param["bottom"] = ["pool2"]
    drop1_param["top"]    = ["drop1"]
    drop1_param["keep_rate"] = 0.25
    params.append(drop1_param)

    # fc1
    fc1_param = {}
    fc1_param["name"] = "FC1"
    fc1_param["type"] = "FCLayer"
    fc1_param["output"] = 256
    fc1_param["bottom"] = ["drop1"]
    fc1_param["top"]    = ["fc1"]
    params.append(fc1_param)

    # fc1_relu
    fc1_relu_param = {}
    fc1_relu_param["name"] = "FC1_RELU"
    fc1_relu_param["type"] = "ReLULayer"
    fc1_relu_param["bottom"] = ["fc1"]
    fc1_relu_param["top"]    = ["fc1_relu"]
    params.append(fc1_relu_param)

    # dropout2
    drop2_param = {}
    drop2_param["name"] = "DROP2"
    drop2_param["type"] = "DropoutLayer"
    drop2_param["bottom"] = ["fc1_relu"]
    drop2_param["top"]    = ["drop2"]
    drop2_param["keep_rate"] = 0.5
    params.append(drop2_param)

    layer5 = {}
    layer5["output"] = 10      # class num
    layer5["name"] = "FC2"
    layer5["type"] = "FCLayer"
    layer5["bottom"] = ["drop2"]
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