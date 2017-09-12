import os
import pdb
from make_net import *
import numpy as np
from load_mnist import *
from net import *
import time
import cv2
import argparse
import sys

def preprocess_images(imgs, mean=127.5, scale=0.0078431):
    new_imgs = []
    for im in imgs:
        im = (im-mean) * scale
        new_imgs.append(im)
    return new_imgs

def load_images(root_dir, ext="jpg"):
    image_set = []
    for root,dirs,files in os.walk(root_dir):
        for f in files:
            if f.find(ext)!=-1:
                image_set.append( os.path.join(root_dir,f) )
    images = []
    for p in image_set:
        img = cv2.imread(p)
        if len(img.shape)==3:
            if img.shape[2]==3:
                img = 0.114*img[:,:,0] + 0.587*img[:,:,0] + 0.299*img[:,:,0]
        images.append(img)
    return images,image_set

def parse_args():
    parser = argparse.ArgumentParser(description='Test a LeNet5 network')
    parser.add_argument('--inference', dest='inference',
                        help='inference or test',
                        default="False", type=str)
    parser.add_argument('--model', dest='saved_model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='image dir', default=None, type=str)
    parser.add_argument('--ext', dest='ext',
                        help='Image extension name',
                        default="jpg", type=str)
    parser.add_argument('--output', dest='output',
                        help='output file name',
                        default="", type=str)
    parser.add_argument('--mean', dest='mean',
                        help='image mean',
                        default=0, type=float)
    parser.add_argument('--scale', dest='scale',
                        help='scale the image',
                        default=0.0078431, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_batch(data, lab, index, batch):
    k = 0
    batch_data = []
    batch_lab  = []
    while index<len(data) and k<batch:
        batch_data.append( data[index] )
        batch_lab.append( lab[index] )
        index += 1
        k += 1
    batch_data = np.array(batch_data)
    batch_lab  = np.array(batch_lab)
    return batch_data,batch_lab,index

def prepare_data_blobs(data_name, batch_data, lab_name,batch_lab):
    batch_data = np.array(batch_data)
    batch_lab   = np.array(batch_lab)
    batch,H,W  = batch_data.shape
    assert H==28 and W==28
    batch_data = batch_data.reshape(len(batch_data), 1, 28, 28)

    data_blob = Blob()
    lab_blob  = Blob()
    data_blob.data_ = batch_data
    lab_blob.data_  = batch_lab

    data_dict = {}
    data_dict[data_name] = data_blob
    data_dict[lab_name]  = lab_blob

    return data_dict

if __name__=="__main__":
    params = {}
    params["model"] = "./LeNet5-3000.model"
    params["data"]  = "data"
    params["label"] = "label"
    params["output"]= "pred"

    #pdb.set_trace()
    args = parse_args()
    if type(args.saved_model)!=type(None):
        params["model"] = args.saved_model
    
    if args.inference=="True":
        data,name_list = load_images(args.image_dir)
        data = preprocess_images(data, args.mean, args.scale)
        lab  = np.random.randint(0,10, len(data))
    elif args.inference=="False":
        mnist_data = load_mnist()
        data = mnist_data["test_data"]
        lab  = mnist_data["test_lab"]
        data = preprocess_images(data, 0.25, 2)
        data = np.array(data).reshape(len(data),28,28)
    else:
        assert 0, "bad args.inference %s"%args.inference
    
    my_net = make_LeNet5_test(params)
    cls_layer = my_net.layers_[-1]
    cls_layer.set_phase("test")

    test_num   = len(data)
    test_batch = 64
    all_pred = []
    all_label= []
    index = 0
    
    while index<test_num:
        t1 = time.time()
        batch_data, batch_lab, index = get_batch(data,lab,index,test_batch)
        data_dict = prepare_data_blobs(params["data"],batch_data,\
                                       params["label"], batch_lab)

        my_net.forward(data_dict)
        pred = my_net.data_blobs_[params["output"]].data_

        for p in pred:
            all_pred.append(p)
        for l in batch_lab:
            all_label.append(l)
        t2 = time.time()
        print "[test] itr:",index,"cost time:",(t2-t1),"s"
    
    all_pred  = np.array(all_pred)
    all_label = np.array(all_label)
    if "False"==args.inference:
        acc = np.sum(all_pred==all_label) / float(len(all_pred))
        print "[test] acc:",acc

    if args.output!="":
        outfd = open(args.output, "w")
        for k in xrange(len(all_pred)):
            if "True"==args.inference:
                print >>outfd,name_list[k],int(all_pred[k])
            else:
                print >>outfd,int(all_pred[k])
        outfd.close()
        print "write results to",args.output,"done!"

    
