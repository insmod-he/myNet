import os
import pdb
from make_net import make_softmax_net


if __name__=="__main__":
    my_net = make_softmax_net()
    my_net.set_lr(1)
    my_net.set_weight_decay(1e-3)
    show_interval = 100
    
    itr = 0
    all_itr = 200
    while itr<all_itr:
        #pdb.set_trace()
        my_net.forward()
        my_net.backward()
        my_net.update_all_params()
        itr += 1

        if itr%show_interval==0:
            loss = my_net.layers_[-1].top[0].data_
            decay = my_net.get_weight_decay()
            
            decay_loss = 0
            if decay>0:
                for L in self.layers_:
                decay_loss += decay*L.calc_weight_decay()
            loss += decay_loss
            print "[train] itr:",itr,"loss:"