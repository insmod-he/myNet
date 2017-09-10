import os
import pdb
from make_net import make_softmax_net, make_2layer_mlp


if __name__=="__main__":
    my_net = make_2layer_mlp()
    my_net.set_lr(1)
    my_net.set_weight_decay(1e-3)
    my_net.set_momemtum(0.2)
    show_interval = 1000
    
    itr = 0
    all_itr = 10000
    while itr<all_itr:
        #pdb.set_trace()
        my_net.forward()
        my_net.backward()
        my_net.update_all_params()
        itr += 1

        if itr%show_interval==0:
            #pdb.set_trace()
            blob_name = my_net.layers_[-1].top_[0]
            loss  = my_net.forward_blobs_[blob_name].data_
            decay = my_net.get_weight_decay()
            
            decay_loss = 0
            if decay>0:
                for L in my_net.layers_:
                    decay_loss += L.calc_weight_decay()
            loss += decay*decay_loss
            print "[train] itr:",itr,"loss:",loss, "decay:",decay_loss