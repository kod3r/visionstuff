import cv2
import sys
import numpy as np

img = cv2.imread(sys.argv[1])
bimg = img[:, :, 0]
gimg = img[:, :, 1]
rimg = img[:, :, 2]

brightness = 10
contrast = -10
lut = np.arange(256)

if contrast > 0:
    delta = 127.0 * contrast / 100.0
    a = 255.0 / (255.0 - delta*2)
    b = a*(brightness - delta)
else:
    delta = -128.0 * contrast/100.0
    a = (256.0 - delta*2)/255
    b = a*brightness + delta

lut = lut*a + b
lut = lut.astype(np.uint8)
print lut
print lut.shape, lut.dtype

bimg = cv2.LUT(bimg, lut)
gimg = cv2.LUT(gimg, lut)
rimg = cv2.LUT(rimg, lut)

cv2.imshow("img", img)

img[:, :, 0] = bimg
img[:, :, 1] = gimg
img[:, :, 2] = rimg

cv2.imshow("newimg", img)
cv2.waitKey(0)
cv2.imwrite("rebsat.jpg", img)