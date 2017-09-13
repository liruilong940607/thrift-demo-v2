import sys
sys.path.append('./gen-py')
sys.path.append('./gen-py/HumanSeg')

from HumanSeg import HumanSeg
from HumanSeg import DataTransfer

from HumanSeg import ttypes
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import numpy as np
import time
import threading
import queue

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2

host = '166.111.71.14'
port = 9095

timeque = queue.Queue()
sendque = queue.Queue()
mutex = threading.Lock()
recv_count = 0
send_count = 0
NotSending = True
slider = None
BgType = 'blur'
BgIndex = 0

bg_capture = [cv2.VideoCapture('background/1505272736.mp4'), \
    cv2.VideoCapture('background/1505273808.mp4'), \
    cv2.VideoCapture('background/2016_10_22_IMG_4576.mp4'), \
    cv2.VideoCapture('background/IMG_2165.mp4')]

################
## recv thread
################


class ThreadRecv (threading.Thread):  
    def __init__(self, client, parent):
        threading.Thread.__init__(self)
        self.client = client
        self.show_frame = parent.show_frame
        #self.bgimg = cv2.imread('background.jpg') 
        #if cv2.__version__[0]=='3':
        #    self.bgimg = cv2.cvtColor(self.bgimg, cv2.COLOR_BGR2RGB) #opencv3  
        #else:
        #    self.bgimg = cv2.cvtColor(self.bgimg, cv2.cv.CV_BGR2RGB)       

    def setsize(self, width, height):
        self.width = width
        self.height = height
        #self.bgimg = cv2.resize(self.bgimg, (width, height))

    def run(self):
        print ('thread start')
        thread_start = time.time()
        global recv_count
        global NotSending
        global slider
        while(True):
            while (send_count == 0):
                print ('send_count:', send_count, 'waiting')
                self.show_frame.clear()
                time.sleep(0.5)
            print ("client - recv", recv_count)
            msg = self.client.recv_bg_blur()
            start = time.time()
            recv_count += 1
            # nparr = np.fromstring(msg.image, np.uint8)
            # if cv2.__version__[0]=='3':
            #     self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #opencv3  
            # else:
            #     self.image = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
            # self.image = cv2.resize(self.image, (self.width, self.height))
            self.image = sendque.get()
            nparr = np.fromstring(msg.blur, np.uint8)
            if cv2.__version__[0]=='3':
                self.seg_fine = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) / 255.0 #opencv3  
            else:
                self.seg_fine = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_UNCHANGED) / 255.0
            #self.seg_fine = self.sigmoid(self.seg_fine)
            self.seg_fine = self.seg_fine[..., np.newaxis]
            self.seg_fine = np.concatenate((self.seg_fine,self.seg_fine,self.seg_fine), axis=2)
            self.seg_fine = cv2.resize(self.seg_fine, (self.width, self.height))
            self.seg_fine = np.float32(self.seg_fine)
            
            global BgIndex
            mutex.acquire()
            success, self.bgimg = bg_capture[BgIndex].read()
            if not success:
                bg_capture[BgIndex].set(cv2.CAP_PROP_POS_FRAMES, 1)
                success, self.bgimg = bg_capture[BgIndex].read()
            mutex.release()
            if cv2.__version__[0]=='3':
                self.bgimg = cv2.cvtColor(self.bgimg, cv2.COLOR_BGR2RGB) #opencv3  
            else:
                self.bgimg = cv2.cvtColor(self.bgimg, cv2.cv.CV_BGR2RGB)
            self.bgimg = cv2.resize(self.bgimg, (self.width, self.height)) 

            if BgType=='blur':
                blurimg = cv2.blur(self.image, (slider.value(),slider.value()))
                blur = np.uint8(self.image*(self.seg_fine)+blurimg*(1-self.seg_fine))
            else:
                blur = np.uint8(self.image*(self.seg_fine)+self.bgimg*(1-self.seg_fine))
            self.nextFrameSlot(blur)
            ## write
            #if cv2.__version__[0]=='3':
            #   output = cv2.cvtColor(np.hstack((self.image,blur)), cv2.COLOR_BGR2RGB)  #opencv3  
            #else:
            #   output = cv2.cvtColor(np.hstack((self.image,blur)), cv2.cv.CV_BGR2RGB)  
            #cv2.imwrite('./res/%d.jpg'%recv_count,output)
            print ("client - recv", recv_count-1, 'done', (time.time()-start)*1000, 'ms')

        totaltime = (time.time()-thread_start)*1000
        print ('thread finished. time: ', totaltime, 'ms')
        print ('thread finished. recv count: ', recv_count)
        print ('thread finished. aver time: ', totaltime/recv_count, 'ms')

    def nextFrameSlot(self, frame):
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        img = img.scaled(self.show_frame.width(), self.show_frame.height(), Qt.KeepAspectRatio)#
        pix = QPixmap.fromImage(img)
        self.show_frame.setPixmap(pix)
        start = timeque.get()
        print ('[Time Delay]', (time.time()-start)*1000, 'ms')

    def sigmoid(self, mask): # [0,1] float ,3 channel
        temp = (mask-0.5)*2*5
        temp = 1.0/(1.0+np.exp(-temp))
        return temp

#####################
## pyQt5
#####################

def resize_nopadding(image, dstshape): # dstshape = [128, 224]
    height = image.shape[0]
    width = image.shape[1]
    ratio = float(width)/height # ratio = (width:height)
    dst_width = int(min(dstshape[1]  * ratio, dstshape[0] ))
    dst_height = int(min(dstshape[0]  / ratio, dstshape[1] ))
    #print ('[padding]: (w,h) =(',width,height,')==>(',dst_width,dst_height,')')
    image_resize = cv2.resize(image, (dst_width, dst_height))
    return image_resize


class VideoCapture(QWidget):
    def __init__(self, filename, parent, recvthread):
        print ('init')
        super(QWidget, self).__init__()
        self.video_frame = parent.video_frame
        if filename=='capture':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(filename[0])  
        timeinit = time.time()      
        success, frame = self.cap.read()
        if cv2.__version__[0]=='3':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#opencv3  
        else:
            frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        img = img.scaled(self.video_frame.width(), self.video_frame.height(), Qt.KeepAspectRatio)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)
        #imgsend = cv2.resize(frame,(368, 640)) # DO NOT change this size
        #reqmsg = ttypes.MSG()
        #reqmsg.image = cv2.imencode('.jpg', imgsend)[1].tostring()

        self.framecost = (time.time()-timeinit)*1000
        print ('time cost each frame in init: ', (time.time()-timeinit)*1000, 'ms')
        recvthread.setsize(self.width, self.height)

        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        print ('init done' )

    def setclient(self, client):
        self.client = client

    def nextFrameSlot(self):
        print ('nextFrameSlot')
        global send_count
        timestart = time.time()
        success, frame = self.cap.read()
        frame = self.light(frame)
        if not success:
            self.release()
            return
        if cv2.__version__[0]=='3':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#opencv3 
        else:
            frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        img = img.scaled(self.video_frame.width(), self.video_frame.height(), Qt.KeepAspectRatio)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)
        timeque.put(time.time())
        if NotSending==False:
            #print 'client - send', send_count
            sendque.put(frame)
            #imgsend = cv2.resize(frame,(0, 0), fx = 0.3, fy = 0.3) # DO NOT change this size
            imgsend = resize_nopadding(frame, [224,224])
            reqmsg = ttypes.MSG()
            reqmsg.image = cv2.imencode('.jpg', imgsend)[1].tostring()
            self.client.send_bg_blur(reqmsg)
            send_count += 1

    def start(self):
        self.timer.start(1000.0/30)

    def release(self):
        self.timer.stop()
        self.cap.release()
        self.video_frame.clear()
        global send_count
        send_count = 0

    def light(self, image):
        value = 60
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = np.array(hsv_image, dtype=np.float32)
        hsv_image[:,:,2] += value
        hsv_image[hsv_image>255] = 255
        hsv_image[hsv_image<0] = 0
        hsv_image = np.array(hsv_image, dtype=np.uint8)
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image
    


class VideoDisplayWidget(QWidget):
    def __init__(self,parent):
        super(VideoDisplayWidget, self).__init__(parent)

        self.startButton = QAction("&LoadCamera", self)
        self.startButton.setShortcut("Ctrl+S")
        self.startButton.setStatusTip('start capture')
        self.startButton.triggered.connect(parent.startCapture)

        self.startVidButton = QAction("&LoadVideo", self)
        self.startVidButton.setShortcut("Ctrl+V")
        self.startVidButton.setStatusTip('Open .h264 File')
        self.startVidButton.triggered.connect(parent.startVideo)

        self.toolMenu = parent.mainMenu.addMenu('&Tools')
        self.toolMenu.addAction(self.startButton)
        self.toolMenu.addAction(self.startVidButton)

        self.layout = QGridLayout(self)
        self.layout.setRowStretch(0, 1)
        self.layout.setRowStretch(1, 10)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.video_frame = QLabel()
        self.show_frame = QLabel()
        self.layout.addWidget(self.createSliderGroup(), 0, 0)
        self.layout.addWidget(self.video_frame, 1, 0)
        self.layout.addWidget(self.show_frame, 1, 1)
        self.setLayout(self.layout)

    def resizeEvent(self, event):
        print ('resize in VideoDisplayWidget')
        return super(VideoDisplayWidget, self).resizeEvent(event) 

    def createSliderGroup(self):
        groupBox = QGroupBox("Option: Background Replace / Blur")

        replacebutton1 = QRadioButton("&Replace1")
        replacebutton1.clicked.connect(lambda: self.changeToReplace(1))
        replacebutton2 = QRadioButton("&Replace2")
        replacebutton2.clicked.connect(lambda: self.changeToReplace(2))
        replacebutton3 = QRadioButton("&Replace3")
        replacebutton3.clicked.connect(lambda: self.changeToReplace(3))
        replacebutton4 = QRadioButton("&Replace4")
        replacebutton4.clicked.connect(lambda: self.changeToReplace(4))
        replacebutton5 = QRadioButton("&Replace5")
        replacebutton5.clicked.connect(lambda: self.changeToReplace(5))

        blurbutton = QRadioButton("&Blur")
        blurbutton.clicked.connect(self.changeToBlur)
        blurbutton.setChecked(True)

        global slider
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        #slider.setTickPosition(QSlider.TicksBothSides)
        #slider.setTickInterval(10)
        slider.setSingleStep(1)
        slider.setRange(1, 101)
        slider.setValue(51)

        vbox = QVBoxLayout()
        replacebox = QWidget()
        replaceboxlayout = QHBoxLayout()
        replaceboxlayout.addWidget(replacebutton1)
        replaceboxlayout.addWidget(replacebutton2)
        replaceboxlayout.addWidget(replacebutton3)
        replaceboxlayout.addWidget(replacebutton4)
        #replaceboxlayout.addWidget(replacebutton5)
        replacebox.setLayout(replaceboxlayout)
        vbox.addWidget(replacebox)
        vbox.addWidget(blurbutton)
        vbox.addWidget(slider)
        groupBox.setLayout(vbox)

        return groupBox

    def changeToReplace(self, index):
        global BgType 
        global BgIndex
        BgType = 'replace'
        mutex.acquire()
        BgIndex = index - 1
        bg_capture[BgIndex].set(cv2.CAP_PROP_POS_FRAMES, 1)
        mutex.release()
        print (BgType)

    def changeToBlur(self):
        global BgType 
        BgType = 'blur'
        print (BgType)

class ControlWindow(QMainWindow):
    def __init__(self):
        super(ControlWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("SegHuman")

        self.capture = None
        self.segment = None
        self.client = None
        self.videoFileName = None
        self.thread_recv = None

        # self.quitAction = QAction("&Exit", self)
        # self.quitAction.setShortcut("Ctrl+Q")
        # self.quitAction.setStatusTip('Close The App')
        # self.quitAction.triggered.connect(self.closeApplication)

        # self.openVideoFile = QAction("&Open Video File", self)
        # self.openVideoFile.setShortcut("Ctrl+Shift+V")
        # self.openVideoFile.setStatusTip('Open .h264 File')
        # self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        # self.fileMenu = self.mainMenu.addMenu('&File')
        # self.fileMenu.addAction(self.openVideoFile)
        # self.fileMenu.addAction(self.quitAction)

        self.videoDisplayWidget = VideoDisplayWidget(self)

        try:
            self.transport = TSocket.TSocket(host, port) # server
            self.transport = TTransport.TBufferedTransport(self.transport)
            self.protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
            self.client = HumanSeg.Client(self.protocol)
            self.transport.open()

            print ("client - try to connect")
            print ("server - " + self.client.try_to_connect())
            print ("client - try to init")
            print ("server - " + self.client.init_image_size(224, 224))
        except (Thrift.TException, ex):
            print ("%s" % (ex.message))

        self.thread_recv = ThreadRecv(self.client, self.videoDisplayWidget)
        self.thread_recv.setDaemon(True)
        self.thread_recv.start()

        self.setCentralWidget(self.videoDisplayWidget)
        print ('init done')

    def resizeEvent(self, event):
        print ('resize in window')
        return super(ControlWindow, self).resizeEvent(event)

    def startCapture(self):
        print ('capture start')
        if self.capture:
            print ('capture release start')
            self.capture.release()
            print ('capture release')
        self.capture = VideoCapture('capture', self.videoDisplayWidget, self.thread_recv)
        self.capture.setclient(self.client)
        self.capture.start()
        global NotSending
        NotSending=False
        self.videoDisplayWidget.startButton.setEnabled(False)
        self.videoDisplayWidget.startVidButton.setEnabled(False)


    def startVideo(self):
        print ('video start')
        self.videoFileName = QFileDialog.getOpenFileName(self, 'Select .h264 Video File')
        if not self.videoFileName[0]=='':
            if self.capture:
                self.capture.release()
                print ('release done')
            self.capture = VideoCapture(self.videoFileName, self.videoDisplayWidget, self.thread_recv)
            self.capture.setclient(self.client)
            self.capture.start()
            global NotSending
            NotSending=False
        self.videoDisplayWidget.startButton.setEnabled(False)

    def closeApplication(self):
        choice = QMessageBox.question(self, 'Message','Do you really want to exit?',QMessageBox.Yes | QMessageBox.No)
        if choice == QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ControlWindow()
    window.showMaximized()
    sys.exit(app.exec_())
