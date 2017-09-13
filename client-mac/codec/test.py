import codec, cv2
import numpy as np
import time

cv2.namedWindow("x")
cap = cv2.VideoCapture(0)
while True:
	beg = time.time()
	frame = cv2.resize(cap.read()[1], (386, 600))
	frame = cv2.flip(frame, 1)
	str = codec.encode(frame)
	t1 = time.time()
	if not str:
		continue
	frame = codec.decode(str)
	t2 = time.time()
	print('encode: %fms %fKB; decode: %fms' % ((t1 - beg) * 1000, len(str) / 1000, (t2 - t1) * 1000))
	cv2.imshow("x", frame)
	cv2.waitKey(1)
