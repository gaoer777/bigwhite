import os.path
import cv2 as cv
import numpy as np

num = 204
path = "./dataset/Dataset220324/FDA"
big_im = cv.imread("./dataset/Dataset220324/FDA/203.png", 0)
for i in range(7):
    im_path = os.path.join(path, str(num+i) + ".png")
    im = cv.imread(im_path, 0)
    big_im = np.concatenate((big_im, im), axis=0)

cv.imwrite("test.png", big_im)
