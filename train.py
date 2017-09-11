import os
import pdb
from make_net import *
import numpy as np
from load_mnist import *
import time

if __name__=="__main__":
    lr = 0.01
    my_net = make_LeNet()
    my_net.set_lr(lr)
    my_net.set_weight_decay(0.0005)
    my_net.set_momemtum(0.9)
    show_interval = 20
    test_interval = 1000
    mnist_data = load_mnist()
    
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
            label = my_net.data_blobs_[lab_blob_name].data_
            acc = (np.argmax(prob,1)==label).sum()/ float(len(prob))

            blob_name = my_net.layers_[-1].top_[0]
            loss  = my_net.data_blobs_[blob_name].data_
            decay = my_net.get_weight_decay()
            
            decay_loss = 0
            if decay>0:
                for L in my_net.layers_:
                    decay_loss += L.calc_weight_decay()
            loss += decay*decay_loss
            print "[train] itr:",itr,"loss: %.2f"%loss, "decay: %.2f"%decay_loss, "acc: %.4f"%acc

        if itr%test_interval==0:
            pdb.set_trace()
            test_batch = 500
            test_num = 10000

            data_layer  = my_net.layers_[0]
            train_batch = data_layer.get_batch()
            data_layer.set_batch(test_batch)
            data_layer.set_phase("test")
            cls_layer = my_net.layers_[-1]
            cls_layer.set_phase("test")
            
            all_pred = []
            all_label= []
            for k in xrange(test_num/test_batch):
                t1 = time.time()
                my_net.forward()
                pred_name = cls_layer.top_[0]
                lab_name  = data_layer.top_[1]
                pred = my_net.data_blobs_[pred_name].data_
                lab  = my_net.data_blobs_[lab_name].data_

                for p in pred:
                    all_pred.append(p)
                for l in lab:
                    all_label.append(l)
                t2 = time.time()
                print "[test] itr:",k,"cost time:",(t2-t1),"s"
            all_pred  = np.array(all_pred)
            all_label = np.array(all_label)
            acc = (all_pred==all_label).sum() / float(len(all_pred))
            print "[test] itr:",itr,"acc:",acc

            data_layer.set_batch(train_batch)
            data_layer.set_phase("train")
            cls_layer.set_phase("train")
        
        