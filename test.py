import os.path
import cv2 as cv
import numpy as np
import ObjectDetect0412

model = ObjectDetect0412.ODAB()
total = sum([param.nelement() for param in model.parameters()])
print(total)

