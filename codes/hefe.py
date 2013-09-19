import cv2
import sys
import numpy as np
img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.cv.CV_BGR2HSV)
h = hsv[:, :, 0]
s = hsv[:, :, 1]
l = hsv[:, :, 2]
cv2.imshow("old hsv", hsv)
print h
Hlut = np.arange(256)
Slut = np.arange(256)
Llut = np.arange(256)
Hval = 100
Sval = 128
Lval = 100
#Hlut[:] = Hval
#Slut[:] = Sval
#Llut[:] = Lval
Hlut[:Hval] = Hlut[:Hval] + abs(Hlut[:Hval]-Hval)**0.92
Hlut[Hval:] = Hlut[Hval:] - abs(Hlut[Hval:]-Hval)**0.92
midh = cv2.LUT(h, Hlut.astype(np.uint8))
hsv[:, :, 0] = midh
newimg = cv2.cvtColor(hsv, cv2.cv.CV_HSV2BGR)
cv2.imshow("only H", newimg)
#Hlut[:] = Hval

Slut[:Sval] = Slut[:Sval] + abs(Slut[:Sval]-Sval)**0.6
Slut[Sval:] = Slut[Sval:] - abs(Slut[Sval:]-Sval)**0.6
Llut[:Lval] = Llut[:Lval] + abs(Llut[:Lval]-Lval)**0.6
Llut[Lval:] = Llut[Lval:] - abs(Llut[Lval:]-Lval)**0.6
h = cv2.LUT(h, Hlut.astype(np.uint8))
s = cv2.LUT(s, Slut.astype(np.uint8))
l = cv2.LUT(l, Llut.astype(np.uint8))

hsv[:, :, 0] = h
hsv[:, :, 1] = s
hsv[:, :, 2] = l
print h
print s
print l
cv2.imshow("new hsv", hsv)
newimg = cv2.cvtColor(hsv, cv2.cv.CV_HSV2BGR)
cv2.imshow("img", img)
cv2.imshow("newimg", newimg)
finimg = cv2.addWeighted(img, 0.4, newimg, 0.6, 0)
cv2.imshow("fin", finimg)
cv2.waitKey(0)
cv2.imwrite("hefe.jpg", finimg)

