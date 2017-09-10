#import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
import pdb

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
        
        self.data_X_ = X
        self.data_y_ = y
        self.data_num_ = len(X)
        self.rnd_idx_  = range(self.data_num_)
        np.random.shuffle(self.rnd_idx_)

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
    
    def updata_param(self, lr):
        pass

class FCLayer():
    def init(self, params):
        #pdb.set_trace()
        self.type_ = "FCLayer"
        self.name_ = params["name"]
        self.output_ = params["output"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        self.W_ = None
        self.b_ = None
        print "[FCLayer] Setup",self.name_

    # bottom[0] -> input data(batch, dim)
    # top[0] -> output data(batch, output)
    def forward(self, bottom, top):
        #pdb.set_trace()
        assert 1==len(bottom)
        assert 1==len(top)

        data = bottom[0].data_
        batch, dim = data.shape
        if None==self.W_:
            self.W_ = 0.01 * np.random.randn(dim, self.output_)
            self.b_ = np.zeros((1,self.output_))

        # forward
        activ = np.dot(data, self.W_) + self.b_
        top[0].data_ = copy.deepcopy(activ)
        self.activ_  = activ

    def backward(self, bottom, top):
        #pdb.set_trace()
        assert 1==len(bottom)
        assert 1==len(top)

        data = bottom[0].data_     # not sure
        grad_top = top[0].diff_
        dW = np.dot(data.T, grad_top)
        db = np.sum(grad_top, 0, keepdims=True)
        dhidden = np.dot(grad_top, self.W_.T)
        bottom[0].diff_ = dhidden

        # weight decay
        self.dW_ = copy.deepcopy(dW)
        self.db_ = copy.deepcopy(db)

    def updata_param(self, lr):
        self.W_ += -1*lr*self.dW_
        self.b_ += -1*lr*self.db_
            
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

        top[0].data_ = np.max(0, bottom[0].data_)

    def backward(self, bottom, top):
        assert 1==len(bottom)
        assert 1==len(top)
        top_diff = copy.deepcopy(top[0].diff_)
        top_diff[bottom[0].data_<=0] = 0.0
        bottom[0].diff_ = top_diff

    def updata_param(self, lr):
        pass


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
        self.prob_= prob

        print "[SoftmaxLossLayer] softmax loss:",data_loss

    def backward(self, bottom, top):
        assert 2==len(bottom)
        #pdb.set_trace()
        batch, dim = bottom[0].data_.shape  # unnormalized score
        
        label = bottom[1].data_
        grad  = copy.deepcopy(self.prob_)
        grad[range(batch), label] -= 1.0
        grad /= batch # ???
        bottom[0].diff_ = grad

    def updata_param(self, lr):
        pass


class L2LossLayer():
    def init(self, params):
        self.type_ = "L2LossLayer"
        self.name_ = params["name"]
        self.bottom_ = params["bottom"]
        self.top_    = params["top"]
        print "[L2LossLayer] Setup",self.name_

    def forward(self):
        pass

    def backward(self):
        pass
    
    def updata_param(self, lr):
        pass

# create layers
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


