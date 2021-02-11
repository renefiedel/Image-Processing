# -*- coding: utf-8 -*-
# By：iloveluoluo
# 2019.3.25

import cv2 as cv
import numpy as np


# Gaussian Blur: Gaussian filter is a linear smoothing low-pass filter, suitable for eliminating Gaussian noise, and is widely used in the noise reduction process of image processing.
# Filtering Gaussian is the process of weighted averaging the entire image. The value of each pixel is obtained by weighted average of itself and other pixel values ​​in the neighborhood.
# Use a template (or convolution, mask) to scan each pixel in the image, and use the weighted average gray value of the pixels in the neighborhood determined by the template to replace the value of the center pixel of the template.


def clamp(pv):
    """Prevent overflow"""


if pv > 255:
    return 255
elif pv < 0:
    return 0
else:
    return pv


def gaussian_noise_demo(image):
    """Add Gaussian noise"""


h, w, c = image.shape
for row in range(0, h):
    for col in range(0, w):
        s = np.random.normal(0, 15, 3)  # Generate random numbers, three each time
    b = image[row, col, 0]  # blue
    g = image[row, col, 1]  # green
    r = image[row, col, 2]  # red
    image[row, col, 0] = clamp(b + s[0])
    image[row, col, 1] = clamp(g + s[1])
    image[row, col, 2] = clamp(r + s[2])
cv.imshow("noise img", image)

src = cv.imread('E:/MyFile/Picture/date/lenademo.png')  # read salt and pepper noise picture
cv.imshow("src demo", src)

# tim1 = cv.getCPUTickCount()
# gaussian_noise_demo(src)
# tim2 = cv.getCPUTickCount()
# time = (tim2-tim1)/cv.getTickFrequency()*1000
# print("time: %s ms" % time)

# Gaussian blur suppresses Gaussian noise
gaussian_noise_demo(src)
# Here (5, 5) means that the length and width of the Gaussian matrix are both 5, and when the standard deviation is 0,
# OpenCV will calculate itself according to the size of the Gaussian matrix, just set one of the two parameters.
# Generally speaking, the larger the size of the Gaussian matrix, the larger the standard deviation, and the greater the degree of blurring of the processed image.
dst = cv.GaussianBlur(src, (5, 5), 0)
cv.imshow("gaussian blur img", dst)

cv.waitKey(0)
cv.destroyAllWindows()