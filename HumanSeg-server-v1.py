import socket
import sys, cv2
sys.path.append('./gen-py')

from HumanSeg import HumanSeg
from HumanSeg import DataTransfer
from HumanSeg.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import time
import numpy as np
import fcn_seg_infer
from datetime import datetime

class HumanSegHandler:
    def try_to_connect(self):
        #global server
	#server.stopped = True
	return "connect successful!"
    
    def try_to_disconnect(self):
        return "disconnect successful!"
    
    def init_image_size(self, width, height):
        net_width = 192
        net_height = 320        
        fcn_seg_infer.net.blobs['data'].reshape(1, 3, net_height, net_width)
        fcn_seg_infer.net.blobs['data'].data[...] = np.zeros((3, net_height, net_width), np.float32)
        fcn_seg_infer.net.forward()
        
        fcn_seg_infer.net2_width = 376
        fcn_seg_infer.net2_height = 600
        fcn_seg_infer.net2.blobs['data'].reshape(1, 3, fcn_seg_infer.net2_height, fcn_seg_infer.net2_width)
        fcn_seg_infer.net2.blobs['data'].data[...] = np.zeros((3, fcn_seg_infer.net2_height, fcn_seg_infer.net2_width), np.float32)
        fcn_seg_infer.net2.blobs['comask'].reshape(1, 1, fcn_seg_infer.net2_height, fcn_seg_infer.net2_width)
        fcn_seg_infer.net2.blobs['comask'].data[...] = np.zeros((1, fcn_seg_infer.net2_height, fcn_seg_infer.net2_width), np.float32)
        fcn_seg_infer.net2.forward()
        
        return 'init image size done!'
    
    def bg_blur(self, msg):
        print 'server--blur'
        timestamp = datetime.now()
        nparr = np.fromstring(msg.image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
        seg_fine = fcn_seg_infer.pred_overall(frame)*255.0
	#blur = fcn_seg_infer.blur2(frame, seg_fine)
        #cv2.imwrite('test_server.jpg',blur)
        rspmsg = MSG()
        rspmsg.image = msg.image
        #rspmsg.image = cv2.imencode('.jpg', seg_fine)[1].tostring()
        rspmsg.blur = cv2.imencode('.jpg', seg_fine)[1].tostring()
        fcn_seg_infer.printTime('[Time] decode+encode+pred', timestamp)
        return rspmsg

    
handler = HumanSegHandler()
processor = HumanSeg.Processor(handler)
transport = TSocket.TServerSocket("166.111.71.14", 9095)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()
 
server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

while True:
	print "Starting thrift server in python..."
	server.serve()
	print "server stop!"
