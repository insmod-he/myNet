import numpy as np
import os
from net import *
from layer import *
import copy

# ok!
def check_SoftmaxLossLayer():
    param = {}
    param["class_num"] = 10
    param["name"]   = "SOFTMAXLOSS"
    param["type"]   = "SoftmaxLossparam"
    param["bottom"] = ["score", "label"]
    param["top"]    = ["loss"]
    
    num = 32
    dim = 99
    L = create_SoftmaxLossLayer(param)
    data = np.random.randn(num, dim) # batch=16, dim=10
    lab  = np.random.randint(0, dim, num)
    
    b0 = Blob()
    b1 = Blob()
    b0.data_ = data
    b1.data_ = lab
    bottom = [b0, b1]
    top = [Blob()]
    old_bottom = copy.deepcopy(bottom)

    L.forward(bottom, top)
    loss = top[0].data_.copy()
    L.backward(bottom, top)
    diff = bottom[0].diff_.copy()

    # numeric gradient
    eps = 1e-5
    n_grad = np.zeros(diff.shape, dtype=np.float64)

    for n in xrange(num):
        for d in xrange(dim):
            lbottom = copy.deepcopy(old_bottom)
            lbottom[0].data_[n, d] -= eps
            ltop = [Blob()]
            L.forward(lbottom, ltop)
            lloss = ltop[0].data_
            
            rbottom = copy.deepcopy(old_bottom)
            rbottom[0].data_[n, d] += eps
            rtop = [Blob()]
            L.forward(rbottom, rtop)
            rloss = rtop[0].data_
            
            g = (rloss - lloss) / (2*eps)
            n_grad[n, d] = g
    
    norm1 = np.sqrt(np.sum(diff*diff))
    norm2 = np.sqrt(np.sum(n_grad*n_grad))
    d = diff-n_grad
    normd = np.sqrt(np.sum(d*d))
    error = normd / np.max(norm1, norm2) / num
    print "[SoftmaxLossLayer] error:",error

def check_ReLULayer():
    param = {}
    param["name"]   = "RELU1"
    param["type"]   = "ReLULayer"
    param["bottom"] = ["data"]
    param["top"]    = ["relu1"]
    
    num = 32
    dim = 64
    ReLU = create_ReLULayer(param)
    data = np.random.randn(num, dim)
    
    b0 = Blob()
    b0.data_ = data
    bottom = [b0]
    top    = [Blob()]
    old_bottom = copy.deepcopy(bottom)

    # loss layer
    lab  = np.random.randint(0, dim, num)
    lab_blob = Blob()
    lab_blob.data_ = lab
    param["class_num"] = dim
    param["name"]   = "SOFTMAXLOSS"
    param["type"]   = "SoftmaxLossparam"
    param["bottom"] = ["score", "label"]
    param["top"]    = ["loss"]
    LossLayer = create_SoftmaxLossLayer(param)

    ReLU.forward( bottom, top )
    loss_bottom = [top[0], lab_blob]
    loss_top = [Blob()]
    LossLayer.forward( loss_bottom, loss_top )
    loss_ori = loss_top[0].data_ # loss

    # backward
    LossLayer.backward( loss_bottom, loss_top )
    ReLU.backward( bottom, [loss_bottom[0]] )
    grad_ori = bottom[0].diff_

    # numeric gradient
    eps = 1e-5
    n_grad = np.zeros(grad_ori.shape, dtype=np.float64)

    for n in xrange(num):
        for d in xrange(dim):
            # left
            lbottom = copy.deepcopy(old_bottom)
            lbottom[0].data_[n, d] -= eps
            ltop = [Blob()]
            ReLU.forward( lbottom, ltop )
            lLoss_bottom = [ltop[0], lab_blob]
            lLoss_top = [Blob()]
            LossLayer.forward( lLoss_bottom, lLoss_top )
            lloss = lLoss_top[0].data_

            # right
            rbottom = copy.deepcopy(old_bottom)
            rbottom[0].data_[n, d] += eps
            rtop = [Blob()]
            ReLU.forward( rbottom, rtop )
            rLoss_top = [Blob()]
            LossLayer.forward( [rtop[0], lab_blob],  rLoss_top )
            rloss = rLoss_top[0].data_

            g = (rloss - lloss) / (2*eps)
            n_grad[n, d] = g
    
    norm1 = np.sqrt(np.sum(grad_ori*grad_ori))
    norm2 = np.sqrt(np.sum(n_grad*n_grad))
    d = grad_ori - n_grad
    normd = np.sqrt(np.sum(d*d))
    error = normd / np.max(norm1, norm2) / num
    print "[ReLULayer] error:",error
    
def check_FCLayer():
    num = 16
    dim = 64
    eps = 1e-5

    param = {}
    param["output"] = dim 
    param["name"] = "FC1"
    param["type"] = "FCLayer"
    param["bottom"] = ["data"]
    param["top"]    = ["fc1"]

    fc = create_FCLayer(param)
    data = np.random.randn(num, dim) # batch=16, dim=10
    lab  = np.random.randint(0, dim, num)
    
    b0 = Blob()
    b1 = Blob()
    b0.data_ = data
    b1.data_ = lab
    top = [Blob()]
    bottom = [b0]
    old_bottom = copy.deepcopy(bottom)

    param["class_num"] = dim
    param["name"]   = "SOFTMAXLOSS"
    param["type"]   = "SoftmaxLoss"
    param["bottom"] = ["score", "label"]
    param["top"]    = ["loss"]
    LossLayer = create_SoftmaxLossLayer(param)

    fc.forward( bottom, top )
    W_ori = copy.deepcopy(fc.W_)
    b_ori = copy.deepcopy(fc.b_)

    loss_bottom = [top[0], b1]
    loss_top = [Blob()]
    LossLayer.forward( loss_bottom, loss_top )
    LossLayer.backward( loss_bottom, loss_top )
    fc.backward( bottom, [loss_bottom[0]] )
    grad_ori = copy.deepcopy(bottom[0].diff_) # backward gradient
    dW_ori = copy.deepcopy(fc.dW_)
    db_ori = copy.deepcopy(fc.db_)

    # numeric gradient
    # n_grad = np.zeros(grad_ori.shape, dtype=np.float64)

    # for n in xrange(num):
    #     for d in xrange(dim):
    #         # left
    #         lbottom = copy.deepcopy(old_bottom)
    #         lbottom[0].data_[n, d] -= eps
    #         ltop = [Blob()]
    #         fc.forward( lbottom, ltop )
    #         lLoss_bottom = [ ltop[0], b1 ]
    #         lLoss_top = [Blob()]
    #         LossLayer.forward( lLoss_bottom, lLoss_top )
    #         lloss = lLoss_top[0].data_

    #         # right
    #         rbottom = copy.deepcopy(old_bottom)
    #         rbottom[0].data_[n, d] += eps
    #         rtop = [Blob()]
    #         fc.forward( rbottom, rtop )
    #         rLoss_bottom = [rtop[0], b1]
    #         rLoss_top = [Blob()]
    #         LossLayer.forward( rLoss_bottom, rLoss_top )
    #         rloss = rLoss_top[0].data_

    #         g = (rloss - lloss) / (2*eps)
    #         n_grad[n, d] = g
    
    # norm1 = np.sqrt(np.sum(grad_ori*grad_ori))
    # norm2 = np.sqrt(np.sum(n_grad*n_grad))
    # d = grad_ori - n_grad
    # normd = np.sqrt(np.sum(d*d))
    # error = normd / np.max(norm1, norm2) / num
    # print "[FCLayer] error:",error

    # check W and b
    n_dW = np.zeros(dW_ori.shape, dtype=np.float)
    w_H, w_W = dW_ori.shape
    for h in xrange(w_H):
        for w in xrange(w_W):
            # left
            lW = copy.deepcopy( W_ori )
            lW[h,w] -= eps
            fc.W_    = lW

            lbottom = copy.deepcopy(old_bottom)
            ltop = [Blob()]
            fc.forward( lbottom, ltop )
            lLoss_bottom = [ ltop[0], b1 ]
            lLoss_top = [Blob()]
            LossLayer.forward( lLoss_bottom, lLoss_top )
            lloss = lLoss_top[0].data_

            # right
            rW = copy.deepcopy( W_ori )
            rW[h,w] += eps
            fc.W_    = rW
            rbottom = copy.deepcopy(old_bottom)
            rtop = [Blob()]
            fc.forward( rbottom, rtop )
            rLoss_bottom = [rtop[0], b1]
            rLoss_top = [Blob()]
            LossLayer.forward( rLoss_bottom, rLoss_top )
            rloss = rLoss_top[0].data_

            fc.W_ = W_ori
            g = (rloss - lloss) / (2*eps)
            n_dW[h, w] = g
    
    norm1 = np.sqrt(np.sum(dW_ori*dW_ori))
    norm2 = np.sqrt(np.sum(n_dW*n_dW))
    d = dW_ori - n_dW
    normd = np.sqrt(np.sum(d*d))
    error = normd / np.max([norm1, norm2]) / num
    print "[FCLayer] dW error:",error


def check_PoolingLayer():
    param = {}
    param["name"]   = "Pooling"
    param["type"]   = "PoolingLayer"
    param["bottom"] = ["data"]
    param["top"]    = ["pooling1"]
    
    num = 4
    dim = 3
    size = 8
    PoolingLayer = create_PoolingLayer(param)
    data = np.random.randn(num, dim, size, size)
    
    b0 = Blob()
    b0.data_ = data
    bottom = [b0]
    top    = [Blob()]
    old_bottom = copy.deepcopy(bottom)

    # loss layer
    lab  = np.random.randint(0, dim, [num, dim, size/2, size/2])
    lab_blob = Blob()
    lab_blob.data_ = lab
    param["class_num"] = dim
    param["name"]   = "L2LOSS"
    param["type"]   = "L2LossLayer"
    param["bottom"] = ["pooling1", "label"]
    param["top"]    = ["loss"]
    LossLayer = create_L2LossLayer(param)

    PoolingLayer.forward( bottom, top )
    loss_bottom = [top[0], lab_blob]
    loss_top = [Blob()]
    LossLayer.forward( loss_bottom, loss_top )

    # backward
    LossLayer.backward( loss_bottom, loss_top )
    PoolingLayer.backward( bottom, [loss_bottom[0]] )
    grad_ori = bottom[0].diff_.copy()

    # numeric gradient
    eps = 1e-5
    n_grad = np.zeros(grad_ori.shape, dtype=np.float64)

    for n in xrange(num):
        for d in xrange(dim):
            for h in xrange(size):
                for w in xrange(size):
                    # left
                    lbottom = copy.deepcopy(old_bottom)
                    lbottom[0].data_[n, d, h, w] -= eps
                    ltop = [Blob()]
                    PoolingLayer.forward( lbottom, ltop )
                    lLoss_bottom = [ltop[0], lab_blob]
                    lLoss_top = [Blob()]
                    LossLayer.forward( lLoss_bottom, lLoss_top )
                    lloss = lLoss_top[0].data_

                    # right
                    rbottom = copy.deepcopy(old_bottom)
                    rbottom[0].data_[n, d, h, w] += eps
                    rtop = [Blob()]
                    PoolingLayer.forward( rbottom, rtop )
                    rLoss_top = [Blob()]
                    LossLayer.forward( [rtop[0], lab_blob],  rLoss_top )
                    rloss = rLoss_top[0].data_

                    g = (rloss - lloss) / (2*eps)
                    n_grad[n, d, h, w] = g
    
    norm1 = np.sqrt(np.sum(grad_ori*grad_ori))
    norm2 = np.sqrt(np.sum(n_grad*n_grad))
    d = grad_ori - n_grad
    normd = np.sqrt(np.sum(d*d))
    error = normd / np.max([norm1, norm2]) / num
    print "[poolingLayer] error:",error

def make_debug_LeNet():
    params = []

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

    top_check_dict = {}
    layers = []
    for idx in xrange(len(params)):
        param = params[idx]
        layer_type  = param["type"]
        create_fun = "create_"+layer_type
        L = eval(create_fun)(param)
        layers.append(L)
        for name in L.top_:
            assert not top_check_dict.has_key(name)
            top_check_dict[name] = ""
    return layers

def net_forward(forward_dict, layers):
    for L in layers_:
        bottom = []
        for name in L.bottom_:
            blob = Blob()
            blob.data_ = forward_dict[name]
            bottom.append(bottom)
        
        top = []
        for name in L.top_:
            blob = Blob()
            top.append(blob)

        L.forward(bottom, top)
        for k in xrange(len(top)):
            name = L.top_[k]
            blob = top[k]
            forward_dict[name] = blob

def net_backward(forward_dict, backward_dict, layers):
    for inv_idx in xrange(len(layers_)-1, -1, -1):
        L   = layers[inv_idx]
        if L.type_=="SoftmaxLossLayer" or L.type_=="L2LossLayer":
            top = [Blob()]
        else:
            top = []
            for name in L.top_:
                top.append( backward_dict[name] )
        
        bottom = []
        for name in L.bottom_:
            bottom.append( forward_dict[name] )

        L.backward(bottom, top)
        
        # save result
        for k in xrange(len(bottom)):
            name = L.bottom_[k]
            diff = bottom[k].diff_
            if backward_dict.has_key(name):
                forward_dict[name].diff_ += diff
            else:
                forward_dict[name].diff_ = diff

def check_LeNet():
    layers = make_debug_LeNet()

    batch = 4
    data  = np.random.randn( batch,1,28,28 )
    label = np.random.randint( 0, 10, batch )
    
    data_blob = Blob(); lab_blob  = Blob()
    data_blob.data_ = data_blob; lab_blob.data_  = lab_blob
    
    forward_dict  = {}
    backward_dict = {}
    forward_dict["data"]  = data_blob
    forward_dict["label"] = lab_blob

    # forward
    forward(forward_dict, layers)
    backward(forward_dict, backward_dict, layers)
    grad_ori = backward_dict

if __name__=="__main__":
    check_PoolingLayer()

