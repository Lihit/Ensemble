import cv2
import numpy as np
import random

img = cv2.imread('demo.jpg')
cv2.imshow('test', img)
img[img > (img.min() + (img.max() - img.min()) * 0.2)] = 0
cv2.imshow('test1', img)
cv2.waitKey(0)
