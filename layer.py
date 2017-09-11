#import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import pdb
import cPickle as PKL
from im2col import *

# class Layer():
#     def __init__():
#         self.bottom_ = []
#         self.top_   = []
#         self.blobs_ = []        # paramerers
#         self.blobs_diff_ = []   # paramerers gradient

#     def init(self, params):
#         self.name_ = params["name"]
#         print "Setup",self.name_

#     def forward(self):
#         pass

#     def backward(self):
#         pass
    
#     def updata_param(self, lr):
#         pass

# top[0] -> data
# top[1] -> label
class MnistDataLayer():
    def init(self, params):
        self.type_  = "MnistDataLayer"
        self.name_  = params["name"]
        self.batch_ = params["batch"]
        self.bottom_= params["bottom"]
        self.top_   = params["top"]
        self.index_ = 0
        self.DEBUG_ = False
        
        if not params.has_key("path"):
            N = 100 # number of points per class
            D = 2   # dimensionality
            K = 3   # number of classes
            X = np.zeros((N*K,D)) # data matrix (each row = single example)
            y = np.zeros(N*K, dtype='uint8') # class labels
            
            for j in xrange(K):
                ix = range(N*j,N*(j+1))
                r = np.linspace(0.0,1,N) # radius
                t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
                X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
                y[ix] = j
            
            if self.DEBUG_:
                # lets visualize the data:
                plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
                plt.show()
                #pdb.set_trace()
        else:
            #pdb.set_trace()
            fd = open(params["path"], "r")
            R  = PKL.load(fd)
            fd.close()
            train_data = R[0][0]
            train_lab  = R[0][1]
            val_data   = R[1][0]
            val_lab    = R[1][1]
            test_data  = R[2][0]
            test_lab   = R[2][1]

            X = np.array(train_data)*2 - 0.5
            X = X.reshape(len(X), 1, 28, 28)
            y = np.array(train_lab)
        
        self.data_X_ = X
        self.data_y_ = y
        self.data_num_ = len(X)
        self.rnd_idx_  = range(self.data_num_)
        np.random.shuffle(self.rnd_idx_)
        print "[MnistDataLayer] Setup",self.name_

    def forward(self, bottom, top):
        assert 2==len(top)
        #pdb.set_trace()

        batch_X = []
        batch_y = []
        for k in xrange(self.batch_):
            if self.index_>=self.data_num_:
                self.index_ = 0
                np.random.shuffle(self.rnd_idx_)
            sel_idx = self.rnd_idx_[self.index_]
            batch_X.append( self.data_X_[sel_idx] )
            batch_y.append( self.data_y_[sel_idx] )
            self.index_ += 1
        top[0].data_ = np.array(batch_X)
        top[1].data_ = np.array(batch_y)

    def backward(self, bottom, top):
        pass
    
    def updata_param(self, param):
        pass

    def calc_weight_decay(self):
        return 0

class FCLayer():
    def init(self, params):
        #pdb.set_trace()
        self.type_ = "FCLayer"
        self.name_ = params["name"]
        self.output_ = params["output"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        self.W_ = []
        self.b_ = []
        self.old_Vw_ = 0
        self.old_Vb_ = 0
        self.data_shape_ = []
        print "[FCLayer] Setup",self.name_

    # bottom[0] -> input data(batch, dim)
    # top[0] -> output data(batch, output)
    def forward(self, bottom, top):
        #pdb.set_trace()
        assert 1==len(bottom)
        assert 1==len(top)

        data = bottom[0].data_
        if 4==len(data.shape):
            self.data_shape_ = copy.deepcopy(data.shape)
            data = data.reshape(data.shape[0], -1)

        batch, dim = data.shape
        if []==self.W_:
            #pdb.set_trace()
            fan_in = float(data.shape[1])
            self.W_ = np.sqrt(2.0/fan_in) * np.random.randn(dim, self.output_)
            self.b_ = np.zeros((1,self.output_))

        # forward
        activ = np.dot(data, self.W_) + self.b_
        top[0].data_ = copy.deepcopy(activ)
        self.activ_  = activ

    def backward(self, bottom, top):
        #pdb.set_trace()
        assert 1==len(bottom)
        assert 1==len(top)

        data = bottom[0].data_.copy()     # not sure
        grad_top = top[0].diff_
        if self.data_shape_!=[]:
            data = data.reshape(data.shape[0],-1)
        
        dW = np.dot(data.T, grad_top)
        db = np.sum(grad_top, 0, keepdims=True)
        dhidden = np.dot(grad_top, self.W_.T)

        if self.data_shape_!=[]:
            dhidden = dhidden.reshape(self.data_shape_)
        bottom[0].diff_ = dhidden

        # save gradient
        self.dW_ = copy.deepcopy(dW)
        self.db_ = copy.deepcopy(db)

    def updata_param(self, param):
        lr = param["lr"]
        decay_coef = param["decay_coef"]
        momemtum   = param["momemtum"]

        Vw = momemtum*self.old_Vw_ - lr*self.dW_
        Vb = momemtum*self.old_Vb_ - lr*self.db_
        self.W_ += Vw
        self.b_ += Vb
        self.old_Vw_ = Vw
        self.old_Vb_ = Vb

        if decay_coef>0:
            self.W_ += -1*decay_coef*self.W_
            #print "[FCLayer] norm2(W):", self.calc_weight_decay()
    
    def calc_weight_decay(self):
        return 0.5*np.sum(self.W_*self.W_)
            
class ReLULayer():
    def init(self, params):
        self.type_ = "ReLULayer"
        self.name_ = params["name"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        print "[ReLULayer] Setup",self.name_

    # bottom[0] -> data
    # top[0]    -> after ReLU
    def forward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)

        #pdb.set_trace()
        data = bottom[0].data_.copy()
        data[data<=0] = 0.0
        top[0].data_  = data

    def backward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)
        top_diff = copy.deepcopy(top[0].diff_)
        top_diff[bottom[0].data_<=0] = 0.0
        bottom[0].diff_ = top_diff

    def updata_param(self, param):
        pass
    
    def calc_weight_decay(self):
        return 0


class SoftmaxLossLayer():
    def init(self, params):
        #pdb.set_trace()
        self.type_  = "SoftmaxLossLayer"
        self.name_  = params["name"]
        self.class_num_ = params["class_num"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        print "[SoftmaxLossLayer] Setup",self.name_

    # bottom[0].data_ -> unnormalized prob
    # bottom[1].data_ -> label
    def forward(self, bottom, top):
        #pdb.set_trace()
        assert 1==len(top)
        assert 2==len(bottom)
        data  = bottom[0].data_
        label = bottom[1].data_

        # calc loss
        batch, dim = data.shape
        exp_data  = np.exp(data)
        prob = exp_data / np.sum(exp_data, 1,  keepdims=True) # prob
        data_loss = -np.log( prob[range(batch), label] )
        data_loss = np.sum(data_loss) / batch
        top[0].data_= data_loss
        self.prob_  = prob

        #print "[SoftmaxLossLayer] softmax loss:",data_loss

    def backward(self, bottom, top):
        assert 2==len(bottom)
        #pdb.set_trace()
        batch, dim = bottom[0].data_.shape  # unnormalized score
        
        label = bottom[1].data_
        grad  = copy.deepcopy(self.prob_)
        grad[range(batch), label] -= 1.0
        grad /= batch # the log loss is a averange, so as the gradient!
        bottom[0].diff_ = grad

    def updata_param(self, param):
        pass
    
    def calc_weight_decay(self):
        return 0

class L2LossLayer():
    def init(self, params):
        self.type_ = "L2LossLayer"
        self.name_ = params["name"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        print "[L2LossLayer] Setup",self.name_

    def forward(self, bottom, top):
        assert 2==len(bottom)
        assert 1==len(top)

        data1 = bottom[0].data_.copy()
        data2 = bottom[1].data_.copy()
        data_diff = data1 - data2
        loss  = 0.5*np.sum(data_diff*data_diff)
        top[0].data_ = loss

    def backward(self, bottom, top):
        assert 2==len(bottom)
        assert 1==len(top)

        data1 = bottom[0].data_.copy()
        data2 = bottom[1].data_.copy()
        data_diff = data1 - data2
        bottom[0].diff_ = data_diff
    
    def updata_param(self, param):
        pass
    
    def calc_weight_decay(self):
        return 0

# create layers

class ConvLayer():
    def init(self, params):
        #pdb.set_trace()
        self.type_ = "ConvLayer"
        self.name_ = params["name"]
        self.output_ = params["output"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        self.pad_    = params["pad"]
        self.kernel_H_ = params["kernel_size"]
        self.kernel_W_ = self.kernel_H_
        self.W_ = []    # num, H, W
        self.b_ = []    # num,
        self.old_Vw_ = 0
        self.old_Vb_ = 0
        print "[ConvLayer] Setup",self.name_
    
    def forward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)
        #pdb.set_trace()

        # 1.data -> data_col
        data = bottom[0].data_.copy()
        assert 4==len(data.shape)  # batch,channel,height,width
        batch,dim,data_H,data_W = data.shape
        outH = data_H - self.kernel_H_ + 1 + 2*self.pad_
        outW = data_W - self.kernel_W_ + 1 + 2*self.pad_

        if []==self.W_:
            #pdb.set_trace()
            fan_in = float(dim*self.kernel_H_*self.kernel_W_)
            self.W_ = np.sqrt(2.0/fan_in) * np.random.randn(self.output_, dim, self.kernel_H_, self.kernel_W_)
            self.b_ = np.zeros((self.output_,))

        # chanel*patchH*patchW, batch*outH*outH
        W_reshape   = self.W_.reshape(self.output_, -1).copy()
        data_cols   = im2col( data, self.pad_, self.kernel_H_, self.kernel_W_ )
        output_cols = W_reshape.dot(data_cols)    # kernel_num, batch*outH*outW
        output = output_cols.reshape(self.output_, batch, outH, outW)
        
        #pdb.set_trace()
        B = self.b_.copy()[np.newaxis][np.newaxis][np.newaxis]
        B = B.transpose([3,0,1,2])
        B = np.tile(B, [1, output.shape[1], output.shape[2], output.shape[3]])
        output = output + B
        output = output.transpose([1,0,2,3]) # batch,output,outH,outW
        top[0].data_ = output
    
    # dOut,dW,db
    def backward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)
        #pdb.set_trace()

        dOut = top[0].diff_    # batch,output,outH,outW
        batch,output,outH,outW = dOut.shape
        data = bottom[0].data_.copy()

        # 1.dData  Out = W.dot(Data) + b  W->(num,dim,kH,kW)
        dOut_cols = dOut.transpose(1,0,2,3).reshape(output, -1)  # output,batch*outH*outW
        W_reshape = self.W_.reshape(self.output_, -1).copy() # num,patch_size
        dData_cols= W_reshape.T.dot(dOut_cols)        # patch_size,bath*outH*outW
        dData     = col2im(dData_cols, self.pad_, \
                           self.kernel_H_, self.kernel_W_, outH, outW) # transform back
        bottom[0].diff_ = dData

        # 2.dW,db
        data_cols = im2col( data, self.pad_, self.kernel_H_, self.kernel_W_ )
        dW = dOut_cols.dot( data_cols.T ).reshape(self.W_.shape)
        db = np.sum(dOut, axis=(0,2,3)) # batch,outH,outW
        
        # save gradient
        self.dW_ = copy.deepcopy(dW)
        self.db_ = copy.deepcopy(db)
    
    def updata_param(self, param):
        lr = param["lr"]
        decay_coef = param["decay_coef"]
        momemtum   = param["momemtum"]

        Vw = momemtum*self.old_Vw_ - lr*self.dW_
        Vb = momemtum*self.old_Vb_ - lr*self.db_
        self.W_ += Vw
        self.b_ += Vb
        self.old_Vw_ = Vw
        self.old_Vb_ = Vb

        if decay_coef>0:
            self.W_ += -1*decay_coef*self.W_
    
    def calc_weight_decay(self):
        return 0.5*np.sum(self.W_*self.W_)


class PoolingLayer():
    def init(self, params):
        self.type_ = "PoolingLayer"
        self.name_ = params["name"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        self.kernel_ = 2
        self.stride_ = 2
        assert self.kernel_==self.stride_
        print "[PoolingLayer] Setup",self.name_

    # bottom[0] -> data
    # top[0]    -> after ReLU
    def forward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)

        #pdb.set_trace()
        data = bottom[0].data_.copy()
        batch, dim, H, W = data.shape

        # data_cols
        outH = H/self.stride_
        outW = W/self.stride_
        Ksize = self.kernel_
        X_cols = np.zeros( [Ksize*Ksize, batch*dim*outH*outW], dtype=np.float)
        
        for c in xrange(batch):
            for d in xrange(dim):
                for out_h in xrange(outH):
                    for out_w in xrange(outW):
                        start_y = out_h*self.stride_
                        end_y   = start_y + Ksize
                        start_x = out_w*self.stride_
                        end_x   = start_x + Ksize
                        patch = data[c, d, start_y:end_y, start_x:end_x]
                        X_cols_idx = c*dim*outH*outW + d*outH*outW + out_h*outW + out_w 
                        X_cols[:, X_cols_idx] = patch.reshape(patch.size)
        arg_max = np.argmax(X_cols, 0)
        Out = X_cols[arg_max, range(X_cols.shape[1])]
        Out = Out.reshape( batch, dim, outH, outW )
        top[0].data_ = Out
        self.argmax_ = copy.deepcopy(arg_max)

    def backward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)

        #pdb.set_trace()
        batch,dim,outH,outW = top[0].diff_.shape
        grad_top = top[0].diff_.copy()
        oriH = self.stride_ * outH
        oriW = self.stride_ * outW
        dData = np.zeros([batch, dim, oriH, oriW], dtype=np.float)

        for b in xrange(batch):
            for d in xrange(dim):
                for h in xrange(outH):
                    for w in xrange(outW):
                        col_idx = b*dim*outH*outW + d*outH*outW + h*outW + w
                        max_idx = self.argmax_[col_idx]
                        ori_h = h*self.stride_ + max_idx/self.kernel_
                        ori_w = w*self.stride_ + max_idx%self.kernel_
                        dData[b, d, ori_h, ori_w] =  grad_top[b, d, h, w]
        bottom[0].diff_ = dData.copy()

    def updata_param(self, param):
        pass
    
    def calc_weight_decay(self):
        return 0

def create_L2LossLayer(param_dict):
    L = L2LossLayer()
    L.init(param_dict)
    return L

def create_SoftmaxLossLayer(param_dict):
    L = SoftmaxLossLayer()
    L.init(param_dict)
    return L

def create_FCLayer(param_dict):
    L = FCLayer()
    L.init(param_dict)
    return L

def create_MnistDataLayer(param_dict):
    L = MnistDataLayer()
    L.init(param_dict)
    return L

def create_ReLULayer(param_dict):
    L = ReLULayer()
    L.init(param_dict)
    return L

def create_ConvLayer(param_dict):
    L = ConvLayer()
    L.init(param_dict)
    return L

def create_PoolingLayer(param_dict):
    L = PoolingLayer()
    L.init(param_dict)
    return L
