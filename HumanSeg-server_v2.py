import socket
import sys, cv2
sys.path.append('./gen-py')

from HumanSeg import HumanSeg
from HumanSeg.ttypes import *

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import time
import numpy as np
import fcn_seg_infer_v2 as fcn_seg_infer
from datetime import datetime

from codec import codec

codec_map = {}

class HumanSegHandler:
    def try_to_connect(self, clientid):
        codec_map[clientid] = codec.get_new_codec()
        return "connect successful!"
    
    def try_to_disconnect(self, clientid):
        codec.release_codec(clientid) 
        return "disconnect successful!"
    
    def init_image_size(self, width, height):
        net_width = 224
        net_height = 224        
        fcn_seg_infer.net_new.blobs['data'].reshape(1, 3, net_height, net_width)
        fcn_seg_infer.net_new.blobs['data'].data[...] = np.zeros((3, net_height, net_width), np.float32)
        fcn_seg_infer.net_new.forward()
        
        return 'init image size done!'
    
    def bg_blur(self, msg, clientid):
        print 'server--blur'
        timestamp = datetime.now()
        image = msg.image
        frame = codec.decode(image, codec_map[clientid])
        frame = frame[:,:,::-1]
        seg_fine = fcn_seg_infer.pred_overall(frame)*255.0
        seg_fine = seg_fine[..., np.newaxis]
        seg_fine = np.concatenate((seg_fine, seg_fine, seg_fine), axis=2)
        fcn_seg_infer.printTime('[Time] decode+encode+pred', timestamp)
        reqmsg = MSG()
        reqmsg.image = codec.encode(seg_fine, codec_map[clientid])
        reqmsg.imageid = msg.imageid
        return reqmsg
    
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
