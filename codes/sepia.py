import cv2
import numpy
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img  = cv2.imread(sys.argv[1])
    else:
        cam = cv2.VideoCapture(0)
        status, img = cam.read()

    m_sepia = numpy.asarray([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia = cv2.transform(img, m_sepia)
    sepia = cv2.cvtColor(sepia, cv2.cv.CV_RGB2BGR)
    cv2.imshow('sepia', sepia)
    cv2.imwrite('sepia.png', sepia)
    cv2.waitKey()
    print sepia.shape