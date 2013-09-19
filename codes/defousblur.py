from SimpleCV import *
import numpy as np

img = Image("lenna")
a = np.zeros((30,30), np.uint8)
cv2.rectangle(a, (10,10), (20,20), (1,1,1), -1)
b = Image(a)
c = b.rotate(45).getGrayNumpy()
d = cv2.bitwise_or(a, c)
e = a/float(np.sum(d))
newimg = cv2.filter2D(img.getNumpy(), -1, e)
Image(newimg).show()
time.sleep(3)