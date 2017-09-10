import os
import sys
import numpy as np
import Queue as Que
import copy
from layer import *

class Blob:
    def __init__(self):
        self.data_ = None
        self.diff_ = None

class Net():
    def init(self, layer_params):
        # 1.Setup layers
        self.forward_flag_  = {}
        self.backward_flag_ = {}
        self.forward_blobs_ = {} # save foward activation
        self.backward_blobs_= {} # save backward gradient 

        # call create_[layer_name]_fun
        top_check_dict = {}
        self.layers_ = []
        for idx in xrange(len(layer_params)):
            param = layer_params[idx]
            layer_type  = param["type"]
            create_fun = "create_"+layer_type
            L = eval(create_fun)(param)
            self.layers_.append(L)

            for name in L.top_:
                assert not top_check_dict.has_key(name)
                top_check_dict[name] = ""
        
        self.lr_ = 1e-3
    
    def set_lr(self, lr):
        self.lr_ = lr
    
    def create_empty_blobs(self, num):
        blobs = []
        for k in xrange(num):
            d = Blob()
            blobs.append(d)
        return blobs
    
    def save_result2dict(self, my_dict, blobs, blob_names):
        for k in xrange(len(blobs)):
            blob = blobs[k]
            name = blob_names[k]  # top name is blob name

            assert not my_dict.has_key(name)
            my_dict[name] = blob
    
    def get_blobs_from_dict(self, result_dict, names):
        blobs = []
        for name in names:
            b = copy.deepcopy( result_dict[name] )
            blobs.append(b)
        return blobs
        
    def forward(self):
        # 1.Init forward flags
        self.forward_blobs_ = {}
        for L in self.layers_:
            assert L.name_!="", "assert self.name_!="""
            
            bottom = self.get_blobs_from_dict(self.forward_blobs_, L.bottom_)
            top    = self.create_empty_blobs(len(L.top_))
            L.forward(bottom, top)
            self.save_result2dict(self.forward_blobs_, top, L.top_) # result save to dict
    
    def save_backward_result2dict(self, my_dict, blobs, names):
        for k in xrange(len(blobs)):
            name = names[k]
            blob = blobs[k]
            if my_dict.has_key(name):
                my_dict[name] += blob
            else:
                my_dict[name] = blob
        
    def backward(self):
        # only surport foward network
        self.backward_blobs_ = {}
        for inv_idx in xrange(len(self.layers_)-1, -1, -1):
            L   = self.layers_[inv_idx]
            if L.type_=="SoftmaxLossLayer" or L.type_=="L2LossLayer":
                top = self.create_empty_blobs(len(L.top_))
            else:
                top = self.get_blobs_from_dict(self.backward_blobs_, L.top_)
            bottom = []
            for name in L.bottom_:
                bottom.append( self.forward_blobs_[name] )

            L.backward(bottom, top)
            self.save_backward_result2dict(self.backward_blobs_, bottom, L.bottom_)

        #print "Backward success!"
    
    def update_all_params(self):
        for L in self.layers_:
            L.updata_param(self.lr_)

