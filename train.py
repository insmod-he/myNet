import os
import pdb
from make_net import make_softmax_net


if __name__=="__main__":
    my_net = make_softmax_net()
    my_net.set_lr(0.1)
    
    itr = 0
    all_itr = 1000
    while itr<all_itr:
        pdb.set_trace()
        my_net.forward()
        my_net.backward()
        my_net.update_all_params()
        itr += 1