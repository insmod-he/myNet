import os
import sys
import numpy as np
import Queue as Que
import copy
from layer import *
import pdb
import cPickle as PKL

class Blob:
    def __init__(self):
        self.data_ = None
        self.diff_ = None

class Net():
    def init(self, layer_params, solver_param={}):
        # 1.Setup layers
        self.forward_flag_  = {}
        self.backward_flag_ = {}
        self.forward_blobs_ = {} # save foward activation
        self.backward_blobs_= {} # save backward gradient 
        self.pretrain_model_= ""

        #pdb.set_trace()
        if len(solver_param.keys())>0:
            if solver_param.has_key("model"):
                self.pretrain_model_ = solver_param["model"]

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
        
        if self.pretrain_model_!="":
            self.load_model(self.pretrain_model_)
        
        self.lr_ = 1e-3
        self.decay_coef_ = 0
        self.momemum_ = 0
    
    def set_momemtum(self, m):
        self.momemum_ = m
        
    def set_lr(self, lr):
        self.lr_ = lr
    
    def set_weight_decay(self, coef):
        self.decay_coef_ = coef
    
    def get_weight_decay(self):
        return self.decay_coef_
    
    def create_empty_blobs(self, num):
        blobs = []
        for k in xrange(num):
            d = Blob()
            blobs.append(d)
        return blobs
    
    def save_result2dict(self, my_dict, blobs, blob_names, copy_diff):
        for k in xrange(len(blobs)):
            blob = copy.deepcopy(blobs[k])
            name = blob_names[k]  # top name is blob name
            if not copy_diff:
                blob.diff_ = None
            
            assert not my_dict.has_key(name)
            my_dict[name] = blob
    
    def get_blobs_from_dict(self, result_dict, names, copy_diff=False):
        blobs = []
        for name in names:
            b = copy.deepcopy( result_dict[name] )
            if False==copy_diff:
                b.diff_ = None
            blobs.append(b)
        return blobs
        
    def forward(self):
        # 1.Init forward flags
        #pdb.set_trace()
        self.data_blobs_ = {}
        for L in self.layers_:
            assert L.name_!="", "assert self.name_!="""
            
            bottom = self.get_blobs_from_dict(self.data_blobs_, L.bottom_, True)
            top    = self.create_empty_blobs(len(L.top_))
            L.forward(bottom, top)
            self.save_result2dict(self.data_blobs_, top, L.top_, False ) # result save to dict
    
    def save_backward_result2dict(self, my_dict, blobs, names):
        for k in xrange(len(blobs)):
            name = names[k]
            blob = blobs[k]
            if my_dict.has_key(name):
                if type(my_dict[name].diff_)==type(None):
                    my_dict[name].diff_ = blob.diff_
                else:
                    my_dict[name].diff_ += blob.diff_
            else:
                assert 0
        
    def backward(self):
        # only surport foward network
        for inv_idx in xrange(len(self.layers_)-1, -1, -1):
            L = self.layers_[inv_idx]
            if L.type_=="SoftmaxLossLayer" or L.type_=="L2LossLayer":
                top = self.create_empty_blobs(len(L.top_))
            else:
                top = self.get_blobs_from_dict(self.data_blobs_, L.top_, True )
            bottom = []
            for name in L.bottom_:
                bottom.append( copy.deepcopy(self.data_blobs_[name]) )

            L.backward(bottom, top)
            self.save_backward_result2dict(self.data_blobs_, bottom, L.bottom_ )
    
    def update_all_params(self):
        param = {}
        param["lr"] = self.lr_
        param["decay_coef"] = self.decay_coef_
        param["momemtum"]   = self.momemum_

        for L in self.layers_:
            L.updata_param(param)
    
    def save(self, name):
        param_dict = {}
        for L in self.layers_:
            L.save(param_dict)
        fd = open(name, "w")
        PKL.dump(param_dict, fd)
        fd.close()
        print name,"saved!"
    
    def load_model(self, path):
        param_dict = PKL.load(open(path, "r"))
        for L in self.layers_:
            L.load(param_dict)
        print "load model form",path
