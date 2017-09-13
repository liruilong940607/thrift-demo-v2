import codec, cv2
import numpy as np
import time

cv2.namedWindow("x")
cap = cv2.VideoCapture(0)
codec_map = codec.get_new_codec()
while True:
    beg = time.time()
    frame = cv2.resize(cap.read()[1], (368, 640))
    frame = cv2.flip(frame, 1)
    print ('[1]')
    print (codec_map)
    str = codec.encode(frame,codec_map)
    print ('[2]')
    t1 = time.time()
    if not str:
        continue
    print ('[3]')
    frame = codec.decode(str,codec_map)
    t2 = time.time()
    print('encode: %fms %fKB; decode: %fms' % ((t1 - beg) * 1000, len(str) / 1000, (t2 - t1) * 1000))
    cv2.imshow("x", frame)
    cv2.waitKey(1)
