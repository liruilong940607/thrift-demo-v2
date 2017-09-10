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

BATCHSIZE = 1

################### about network 1 #########################
#INPUT_CHANNEL = 3

Mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
Std = 0.015625

# load net
DEPLOY = release_dir+'/model/coarse/VGG16_1c_dep.prototxt'
CAFFEMODEL = release_dir+'/model/coarse/vgg_co_2labels_minAug_3c_iter_11000.caffemodel'
net = caffe.Net(DEPLOY, CAFFEMODEL, caffe.TEST)
caffe.set_mode_gpu();
caffe.set_device(1);

totalnettime = 0

def predict(img):
    global totalnettime
    imgshape = img.shape
    img = cv2.resize(img,(192,320))
    img = np.float32(img)
    img -= Mean
    img *= Std
    img = img.transpose((2,0,1))
        
    net.blobs['data'].reshape(1,*img.shape)
    net.blobs['data'].data[...] = img

    # run net and take argmax for prediction
    aa = datetime.now()
    net.forward()
    dt = datetime.now()-aa
    print 'net', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
    
    out = net.blobs['mask/prob'].data[0]
    out = out.transpose().reshape((32,18,2))[:,:,1]
    out = cv2.resize(out*255,(imgshape[1],imgshape[0]),interpolation = cv2.INTER_CUBIC)##368 640
    return out

############################################################

################### about network 2 #########################
#INPUT_CHANNEL = 4
MEAN = [114.578, 115.294, 108.353]

# load net

DEPLOY2 = release_dir+'/model/fine/googlenet_seg_fine.prototxt'
CAFFEMODEL2 = release_dir+'/model/fine/googlenet_fine_4c_1l_noWeight_iter_27000.caffemodel'
net2 = caffe.Net(DEPLOY2, CAFFEMODEL2, caffe.TEST)

net2_width = 376
net2_height = 600
        
def predict2(img, cimg):
    aa = datetime.now()
    imgshape = img.shape
    if img.shape[0] != net2_height or img.shape[1]!=net2_width:
        img = cv2.resize(img,(net2_width,net2_height))
        cimg = cv2.resize(cimg,(net2_width,net2_height))
    dt = datetime.now()-aa
    print 'resize', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
    img = np.float32(img)

    img -= Mean
    img *= Std
    img = img.transpose((2,0,1))

    cimg = np.float32(cimg)
    cimg = (cimg-128)*Std
    cm = cimg[np.newaxis, ...]
    
    #cm[0,:,:] = 0
    net2.blobs['data'].reshape(1,*img.shape)
    net2.blobs['data'].data[...] = img
    net2.blobs['comask'].reshape(1,*cm.shape)#
    net2.blobs['comask'].data[...] = cm#

    aa = datetime.now()
    net2.forward()
    dt = datetime.now()-aa
    print 'net2', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0

    out = net2.blobs['mask/prob'].data[0]#
    out = out.transpose((1,2,0))[:,:,1]

    out = cv2.resize(out,(imgshape[1],imgshape[0]),interpolation = cv2.INTER_CUBIC)
    return out

###################### blur method #########################
def blur1(frame, frame_mask2):#0/1
    mapping_width = 15
    frame_blur_pyr = []
    for j in range(0,mapping_width+2):
        frame_blur_pyr.append(cv2.blur(frame, (2*j+1,2*j+1))) 
    #element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #frame_mask2 = cv2.dilate(frame_mask2,element)
    frame_mask2_dila = frame_mask2.copy()
    for j in range(1,mapping_width+1):
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*j+1,2*j+1))
        frame_mask2_dila += cv2.dilate(frame_mask2,element)
    pyr_index = mapping_width+1-frame_mask2_dila

    frame_output = frame.copy()
    for k in range(0,mapping_width+2):
        frame_output[pyr_index==k] = frame_blur_pyr[k][pyr_index==k]
    return frame_output


def blur2(frame, frame_mask2):#0/1
    timestamp = datetime.now()
    frame_blur = cv2.blur(frame, (31,31))
    frame_mask2_bgr = np.float32(frame_mask2)/255.0#cv2.cvtColor(frame_mask2, cv2.COLOR_GRAY2BGR);
    frame_output = frame*frame_mask2_bgr + frame_blur*(1-frame_mask2_bgr)
    printTime('[Time] fcn_seg_infer.blur2 ', timestamp)
    return frame_output


def blur3(frame, frame_mask2_input):#0/1
    frame_blur = cv2.blur(frame, (31,31))
    frame_mask2 = frame_mask2_input.copy()*255
    frame_mask2 = np.array(frame_mask2, dtype=np.float32)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    frame_mask2 = cv2.dilate(frame_mask2,element)
    frame_mask2 = cv2.blur(frame_mask2, (31,31)) 
    frame_mask2_3channel = cv2.cvtColor(frame_mask2, cv2.COLOR_GRAY2BGR);
    
    frame_output = frame*frame_mask2_3channel/255 + frame_blur*(1-frame_mask2_3channel/255)
    return frame_output
############################################################

##########

def predict_img(frame):
    aa = datetime.now()
    caffe.set_mode_gpu();
    caffe.set_device(1);
    dt = datetime.now()-aa
    print 'caffe.set', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
    aa = datetime.now()
    ## predict
    frame_mask = predict(frame)
    dt = datetime.now()-aa
    print 'predict frame', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
    aa = datetime.now()
    out = predict2(frame, frame_mask) ## 0/1
    dt = datetime.now()-aa
    print 'predict2 frame', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
    aa = datetime.now()
    frame_mask2_bgr = cv2.cvtColor(out*255, cv2.COLOR_GRAY2BGR)
    dt = datetime.now()-aa
    print 'cv2Color', (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0
    # ## blur
    # frame_output = blur3(frame, frame_mask2)
    # ## show
    # frame_show = np.zeros([frame.shape[0], frame.shape[1]*3+20*2,3], dtype=np.uint8)
    # frame_show[0:frame.shape[0],0:frame.shape[1],:] = frame
    # frame_show[0:frame.shape[0],frame.shape[1]+20:frame.shape[1]*2+20,:] = frame_output
    # frame_show[0:frame.shape[0],frame.shape[1]*2+40:frame.shape[1]*3+40,:] = (frame_mask2_bgr)
    return frame_mask2_bgr

def scale_img(img):
    if img.shape[0] >= 600 and img.shape[1]>=376:
        ratio_h = 600.0/img.shape[0]
        ratio_w = 376.0/img.shape[1]
        if ratio_h<ratio_w:
            img = cv2.resize(img,(0,0), fx=ratio_w, fy=ratio_w)
        else:
            img = cv2.resize(img,(0,0), fx=ratio_h, fy=ratio_h)
    return img
    

def predict_video(videoname):
    caffe.set_mode_gpu();
    caffe.set_device(1);

    videoCapture = cv2.VideoCapture(videoname)
    success, frame = videoCapture.read() 
    if success:
        frame = scale_img(frame)
    frame_show = np.zeros([frame.shape[0], frame.shape[1]*3+20*2,3])
    videoWriter = cv2.VideoWriter(videoname+'_pred.avi', cv2.cv.FOURCC(*'XVID') , 30, (frame_show.shape[1],frame_show.shape[0])) 
    while success :
        ## batch
        count = 0
        frames = np.ndarray((BATCHSIZE,frame.shape[0],frame.shape[1],3),dtype=np.float32)
        while count < BATCHSIZE and success:
            frames[count,:,:,:] = frame
            count +=1
            success, frame = videoCapture.read() 
            if success:
                frame = scale_img(frame)
        ## predict
        frame_masks = predict_batch(frames)
        outs, frame_mask2s = predict2_batch(frames, frame_masks) ## 0/1
        
        for i in range(0,count):
            out = outs[i,:,:,0]
            frame_mask = frame_masks[i,:,:,0]
            frame_mask2 = frame_mask2s[i,:,:,0]
            
            frame_mask2_bgr = cv2.cvtColor(out*255, cv2.COLOR_GRAY2BGR);
            ## blur
            frame_output = blur3(frames[i,:,:,:], frame_mask2)
            ## show
            frame_show = np.zeros([frames.shape[1], frames.shape[2]*3+20*2,3], dtype=np.uint8)
            frame_show[0:frames.shape[1],0:frames.shape[2],:] = frames[i,:,:,:]
            frame_show[0:frames.shape[1],frames.shape[2]+20:frames.shape[2]*2+20,:] = frame_output
            frame_show[0:frames.shape[1],frames.shape[2]*2+40:frames.shape[2]*3+40,:] = (frame_mask2_bgr)
            ## write
            videoWriter.write(cv2.resize(frame_show, (frame_show.shape[1],frame_show.shape[0])))
    return videoname+'_pred.avi'

def printTime(name, starttime):
    dt = datetime.now()-starttime
    print name, (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds/1000.0 

def pred_overall(frame):
    timestamp = datetime.now()
    caffe.set_mode_gpu();
    caffe.set_device(1);
    # net1
    imgshape = frame.shape
    img = cv2.resize(frame,(192,320))
    img = np.float32(img)
    img -= Mean
    img *= Std
    img = img.transpose((2,0,1))
    #net.blobs['data'].reshape(1,*img.shape)
    net.blobs['data'].data[...] = img

    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['mask/prob'].data[0]
    out = out.transpose().reshape((32,18,2))[:,:,1]
    #cimg = cv2.resize(out*255,(imgshape[1],imgshape[0]),interpolation = cv2.INTER_CUBIC)##368 640
    
    # net2
    if frame.shape[0] != net2_height or frame.shape[1]!=net2_width:
        img = cv2.resize(frame,(net2_width,net2_height))
    cimg = cv2.resize(out*255,(net2_width,net2_height))
    
    img = np.float32(img)
    img -= Mean
    img *= Std
    img = img.transpose((2,0,1))
    
    cimg = np.float32(cimg)
    cimg = (cimg-128)*Std
    cm = cimg[np.newaxis, ...]
    #cm[0,:,:] = 0
    net2.blobs['data'].reshape(1,*img.shape)
    net2.blobs['data'].data[...] = img
    net2.blobs['comask'].reshape(1,*cm.shape)#
    net2.blobs['comask'].data[...] = cm#

    net2.forward()
    out = net2.blobs['mask/prob'].data[0]#
    out = out.transpose((1,2,0))[:,:,1]#*255
    #out = out[..., np.newaxis]
    #out = np.concatenate((out,out,out), axis=2)
    #out = cv2.resize(out,(imgshape[1],imgshape[0]))
    printTime('[Time] fcn_seg_infer.pred_overall ', timestamp)
    return out


# urls = ('/upload', 'Upload',
#         '/download', 'Download')
# render = web.template.render('templates/',)
# DownloadFile = ''
# totaltime = 0
# class Upload:

#     def GET(self):
#         web.header("Content-Type","text/html; charset=utf-8")
#         return render.upload('')

#     def POST(self):
#         x = web.input(myfile={},myvideo={})
#         if not x.myfile=={}: # to check if the file-object is created
#             filedir = './static'
#             filepath=x.myfile.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
#             if not filepath=='':
#                 filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
#                 fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
#                 fout.write(x.myfile.file.read()) # writes the uploaded file to the newly created file.
#                 fout.close() # closes the file, upload complete.

#                 imgname = filedir +'/'+filename
#                 predict_outfile = predict_img(imgname)
#                 return render.upload(predict_outfile)#outfile
        
#         if not x.myvideo=={}: 
#             filedir = './static'
#             filepath=x.myvideo.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
#             if not filepath=='':
#                 filename=filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
#                 fout = open(filedir +'/'+ filename,'wb') # creates the file where the uploaded file should be stored
#                 fout.write(x.myvideo.file.read()) # writes the uploaded file to the newly created file.
#                 fout.close() # closes the file, upload complete.

#                 videoname = filedir +'/'+filename
#                 global totaltime
#                 global totalnettime
#                 totaltime = 0
#                 totalnettime = 0
#                 start = time.clock()
#                 global net 
#                 global net2
                
#                 print 'video process start!', filename
#                 predict_outfile = predict_video(videoname)              
#                 net = caffe.Net(DEPLOY, CAFFEMODEL, caffe.TEST)
#                 net2 = caffe.Net(DEPLOY2, CAFFEMODEL2, caffe.TEST)
                
#                 print 'video process done!', filename
#                 end = time.clock()
#                 totaltime = end-start
#                 global DownloadFile
#                 DownloadFile = predict_outfile.split('/')[-1]
#                 print 'Time cost [CNN]: ', totalnettime/1000, 's'
#                 print 'Time cost [ALL]: ', totaltime, 's'
#                 raise web.seeother('/download')
#                 return render.upload('')#download : http://166.111.71.103:8888/files/webdemo/static/test2.avi_pred.avi
        
#         return render.upload('') 
    
# BUF_SIZE = 262144    
# class Download:
#     def GET(self):
#         file_name = DownloadFile
#         file_path = os.path.join('./static/', file_name)
#         f = None
#         try:
#             f = open(file_path, "rb")
#             web.header('Content-Type','video/avi')
#             web.header('Content-disposition', 'attachment; filename=%s' % file_name)
#             while True:
#                 c = f.read(BUF_SIZE)
#                 if c:
#                     yield c
#                 else:
#                     break
#         except Exception, e:
#             print e
#             yield 'Error'
#         finally:
#             if f:
#                 f.close()

# if __name__ == "__main__":
#     app = web.application(urls, globals()) 
#     app.run()
