import os
import pdb
from make_net import *
import numpy as np

if __name__=="__main__":
    lr = 1e-2
    my_net = make_3layer_mlp()
    my_net.set_lr(lr)
    my_net.set_weight_decay(1e-4)
    my_net.set_momemtum(0.9)
    show_interval = 200
    
    itr = 0
    all_itr = 10000
    while itr<all_itr:
        #pdb.set_trace()
        my_net.forward()
        my_net.backward()
        my_net.update_all_params()
        itr += 1

        if itr>8000:
            my_net.set_lr(lr/10)

        if itr%show_interval==0:
            #pdb.set_trace()
            prob  = my_net.layers_[-1].prob_
            lab_blob_name = my_net.layers_[0].top_[1]
            label = my_net.forward_blobs_[lab_blob_name].data_
            acc = (np.argmax(prob,1)==label).sum()/ float(len(prob))

            blob_name = my_net.layers_[-1].top_[0]
            loss  = my_net.forward_blobs_[blob_name].data_
            decay = my_net.get_weight_decay()
            
            decay_loss = 0
            if decay>0:
                for L in my_net.layers_:
                    decay_loss += L.calc_weight_decay()
            loss += decay*decay_loss
            print "[train] itr:",itr,"loss:",loss, "decay:",decay_loss, "acc:",acc
        