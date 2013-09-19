import cv2
import sys
import numpy as np

class rgbSat:
    def __init__(self, filename):
        self.img = cv2.imread(filename)
        self.origimg = np.copy(self.img)
        self.brightness = 0
        self.contrast = 0
        cv2.namedWindow("newimage", cv2.CV_WINDOW_AUTOSIZE)
        cv2.createTrackbar('brightness', 'newimage', self.brightness, 200, self.changeBrightness)
        cv2.createTrackbar('contrast', 'newimage', self.contrast, 200, self.changeContrast)
        cv2.imshow("newimage", self.img)
        cv2.imshow("image", self.origimg)
        cv2.waitKey(0)

    def changeBrightness(self, arg):
        print arg, "arg"
        self.brightness = arg - 100
        print self.brightness, "brightness"
        print self.contrast, "contrast"

        self.bimg = self.origimg[:, :, 0]
        self.gimg = self.origimg[:, :, 1]
        self.rimg = self.origimg[:, :, 2]

        lut = np.arange(256)

        if self.contrast > 0:
            delta = 127.0 * self.contrast / 100.0
            a = 255.0 / (255.0 - delta*2)
            b = a*(self.brightness - delta)
        else:
            delta = -128.0 * self.contrast/100.0
            a = (256.0 - delta*2)/255
            b = a*self.brightness + delta

        lut = lut*a + b
        lut = lut.astype(np.uint8)

        self.bimg = cv2.LUT(self.bimg, lut)
        self.gimg = cv2.LUT(self.gimg, lut)
        self.rimg = cv2.LUT(self.rimg, lut)

        self.img[:, :, 0] = self.bimg
        self.img[:, :, 1] = self.gimg
        self.img[:, :, 2] = self.rimg

        cv2.imshow("newimage", self.img)

    def changeContrast(self, arg):
        print arg, "arg"
        self.contrast = arg - 100
        print self.brightness, "brightness"
        print self.contrast, "contrast"
        lut = np.arange(256)

        self.bimg = self.origimg[:, :, 0]
        self.gimg = self.origimg[:, :, 1]
        self.rimg = self.origimg[:, :, 2]

        if self.contrast > 0:
            delta = 127.0 * self.contrast / 100.0
            a = 255.0 / (255.0 - delta*2)
            b = a*(self.brightness - delta)
        else:
            delta = -128.0 * self.contrast/100.0
            a = (256.0 - delta*2)/255
            b = a*self.brightness + delta

        lut = lut*a + b
        lut = lut.astype(np.uint8)

        self.bimg = cv2.LUT(self.bimg, lut)
        self.gimg = cv2.LUT(self.gimg, lut)
        self.rimg = cv2.LUT(self.rimg, lut)

        self.img[:, :, 0] = self.bimg
        self.img[:, :, 1] = self.gimg
        self.img[:, :, 2] = self.rimg

        cv2.imshow("newimage", self.img)

rgbSat(sys.argv[1])
