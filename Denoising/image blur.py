# -*- coding: utf-8 -*-
# By：iloveluoluo
# 2019.3.24
import cv2 as cv
import numpy as np

# Fuzzy Operation: Based on discrete convolution, each convolution kernel is defined. Different convolution kernels
# get different convolution effects. Blurring is a form of convolution.

# : The mean value filtering is a typical linear filtering algorithm, which means giving a template to the target
# pixel on the image.
# The template includes adjacent pixels around it (8 pixels around the target pixel, forming a filter template,
# that is, removing the target pixel itself).


# Replace the original pixel value with the average of all the pixels in the template. Since the pixels on the border
# of the image cannot be covered by the template, the edge of the image is missing.

# Median Fuzzy: The median value after sorting the data from small to large, using a 3×3 size template for median
# filtering. Sort the 9 numbers in the template from small to large: 1, 1, 1, 2, 2, 5, 6, 6, 10. The intermediate
# value is 2, and the value of the (2, 2) position after the median filter becomes 2. The same applies to other pixels.

# Custom Blur: (sharpening) is where the image details are highlighted or the image is blurred.


def blur_demo(image):
    """
         Mean blur: de-random noise
         Blur can only define the size of the convolution kernel
    """
    dst_y = cv.blur(image, (1, 10))  # Y direction blur, 1X10 convolution kernel
    dst_x = cv.blur(image, (10, 1))  # X direction blur, 10X1 convolution kernel
    dst_xy = cv.blur(image, (5, 5))  # block ambiguity, 5X5 convolution kernel


cv.imshow("blurY demo", dst_y)
cv.imshow("blurX demo", dst_x)
cv.imshow("blurXY demo", dst_xy)


def median_blur_demo(image):
    """Median Blur: Denoising """
    dst_xy = cv.medianBlur(image, 5)  # neighborhood X*X, X must be the base


cv.imshow("median_blurXY demo", dst_xy)


def custom_blur_demo(image):
    """Custom Blur """
    Kernel1 = np.ones([5, 5], np.float32) / 25  # 5X5 Value 1 convolution kernel, /25 prevents overflow
    # Convolution sharpening operator to increase contrast
    Kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 5X5 convolution kernel


dst1 = cv.filter2D(image, -1, kernel1)
dst2 = cv.filter2D(image, -1, kernel2)
cv.imshow("custom_blur demo1", dst1)
cv.imshow("custom_blur demo2", dst2)

Src = cv.imread('E:/MyFile/Picture/date/lenademo.png')  # Read the salt and pepper noise picture

# blur_demo(src)
# median_blur_demo(src)
custom_blur_demo(src)

cv.imshow("src demo", src)
cv.waitKey(0)
cv.destroyAllWindows()