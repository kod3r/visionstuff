import cv2
import sys
import numpy as np

class hsvTrack:
    def __init__(self, filename):
        self.img = cv2.imread(filename)
        self.origimg = np.copy(self.img)
        self.finimg = np.copy(self.img)
        self.H = 0
        self.S = 128
        self.V = 128
        self.alpha = 40
        self.beta = 60
        cv2.namedWindow("newimage", cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow("finImage", cv2.CV_WINDOW_AUTOSIZE)
        cv2.createTrackbar('H', 'newimage', self.H, 180, self.changeH)
        cv2.createTrackbar('S', 'newimage', self.S, 255, self.changeS)
        cv2.createTrackbar('V', 'newimage', self.V, 255, self.changeV)
        cv2.createTrackbar('alpha', 'finImage', self.alpha, 100, self.changeAlpha)
        cv2.createTrackbar('beta', 'finImage', self.beta, 100, self.changeBeta)
        cv2.imshow("newimage", self.img)
        cv2.imshow("image", self.origimg)
        cv2.imshow("finImage", self.finimg)
        cv2.waitKey(0)

    def change(self):
        hsv = cv2.cvtColor(self.origimg, cv2.cv.CV_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        l = hsv[:, :, 2]

        Hlut = np.arange(256)
        Slut = np.arange(256)
        Llut = np.arange(256)

        Hval = self.H
        Sval = self.S
        Lval = self.V

        Hlut[:Hval] = Hlut[:Hval] + abs(Hlut[:Hval]-Hval)**0.72
        Hlut[Hval:] = Hlut[Hval:] - abs(Hlut[Hval:]-Hval)**0.72
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

        self.img = cv2.cvtColor(hsv, cv2.cv.CV_HSV2BGR)
        self.finimg = cv2.addWeighted(self.origimg, self.alpha/100.0, self.img, self.beta/100.0, 0)
        cv2.imshow("newimage", self.img)
        cv2.imshow("finImage", self.finimg)

    def changeH(self, hval):
        self.H = hval
        self.change()

    def changeS(self, sval):
        self.S = sval
        self.change()

    def changeV(self, vval):
        self.V = vval
        self.change()

    def changeAlpha(self, alpha):
        self.alpha = alpha
        self.change()

    def changeBeta(self, beta):
        self.beta = beta
        self.change()

hsvTrack(sys.argv[1])
