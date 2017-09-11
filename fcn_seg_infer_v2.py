import web
from PIL import Image

##########

import cv2
import numpy as np
from PIL import Image
import sys
release_dir = '/home/dalong/Workspace/human_matting/fcn_seg/release'
sys.path.insert(0,release_dir+'/caffe/python')
import caffe
import os
import time
from datetime import datetime

################### new network #########################

def resize_padding2(image, dstshape): # dstshape = [128, 224]
    height = image.shape[0]
    width = image.shape[1]
    ratio = float(width)/height # ratio = (width:height)
    dst_width = int(min(dstshape[1]  * ratio, dstshape[0] ))
    dst_height = int(min(dstshape[0]  / ratio, dstshape[1] ))
    origin = [int((dstshape[1] - dst_height)/2), int((dstshape[0] - dst_width)/2)]
    #print '[padding]: (w,h) =(',width,height,')==>(',dst_width,dst_height,')'
    if len(image.shape)==3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        newimage = np.zeros(shape = (dstshape[1], dstshape[0], image.shape[2]), dtype = np.uint8)
        newimage[origin[0]:origin[0]+dst_height, origin[1]:origin[1]+dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height),  interpolation = cv2.INTER_NEAREST)
        newimage = np.zeros(shape = (dstshape[1], dstshape[0]), dtype = np.uint8)
        newimage[origin[0]:origin[0]+height, origin[1]:origin[1]+width] = image
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    return newimage, bbx


# load net
net_new = caffe.Net('/home/dalong/Workspace0/experiment/4_mobilenet_trytofix/infer_hair_mobile_nn_v2.prototxt', \
                '/home/dalong/Workspace0/experiment/4_mobilenet_trytofix/snapshot/init03_224_lr04_iter_27000.caffemodel', \
                caffe.TEST)

def predict_new(image): # 224 224
    img, bbx= resize_padding2(image, [224, 224])
    img = np.float32(img)
    img -= np.array((128.0, 128.0, 128.0), dtype=np.float32)
    img = img.transpose((2,0,1))
        
    net_new.blobs['data'].reshape(1,*img.shape)
    net_new.blobs['data'].data[...] = img
    net_new.forward()
    
    out = net_new.blobs['prob'].data[0]
    out = out[1] 
    
    out = out[bbx[1]:bbx[3], bbx[0]:bbx[2]]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))
    return out

def printTime(name, starttime):
    dt = datetime.now()-starttime
    print name, (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0 

def pred_overall(frame):
    timestamp = datetime.now()
    caffe.set_mode_gpu();
    caffe.set_device(1);
    out = predict_new(frame) # [0,1]
    printTime('[Time] fcn_seg_infer.pred_overall ', timestamp)
    return out

