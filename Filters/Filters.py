import imageio
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

image = cv2.imread('/home/renefiedel/PycharmProjects/Image-Processing/Images/Lenna_(test_image).png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(11, 6))
plt.imshow(image)
plt.title('Image')
plt.xticks([])
plt.yticks([])
plt.show()
imageio.imsave('Output/COLOR_BGR2RGB.jpg', image.astype(np.uint8))

#  matplotlib inline
#  A 9 x 9 mean filter will be applied to the image.
image = cv2.imread('/home/renefiedel/PycharmProjects/Image-Processing/Images/Lenna_(test_image).png')  # reads the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
figure_size = 9  # the dimension of the x and y axis of the kernel.
new_image = cv2.blur(image, (figure_size, figure_size))
fig = plt.figure(figsize=(11, 6))
imageio.imsave('Output/COLOR_BGR2HSV.jpg',
               image.astype(np.uint8))
fig.suptitle('MEAN FILTER')
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)), plt.title('Mean filter')
plt.xticks([]), plt.yticks([])

#  plt.show()
plt.savefig('Output/mean_filter.png', bbox_inches="tight")
plt.close(fig)
# grey scale
# The image will first be converted to grayscale
image2 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
figure_size = 9
new_image = cv2.blur(image2, (figure_size, figure_size))
imageio.imsave('Output/COLOR_greyscale.jpg', new_image.astype(np.uint8))

fig = plt.figure(figsize=(11, 6))
fig.suptitle('MEAN FILTER grey')
plt.subplot(121), plt.imshow(image2, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image, cmap='gray'), plt.title('Mean filter')
plt.xticks([]), plt.yticks([])
#  plt.show()
plt.savefig('Output/mean_filter_gray.jpg')
plt.close(fig)

#  ----------------------
#  GAUSSIAN FILTER
#  ----------------------
#  A 9 x 9 Gaussian filter (with sigma = 0) will be applied to the image.
new_image = cv2.GaussianBlur(image, (figure_size, figure_size), 0)

fig = plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)), plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
#  plt.show()
plt.savefig('Output/gaussian_filter_gray.jpg')
plt.close(fig)

#  gray scale gauss
new_image_gauss = cv2.GaussianBlur(image2, (figure_size, figure_size), 0)

fig = plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(image2, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image_gauss, cmap='gray'), plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
#  plt.show()
plt.savefig('Output/Gaussian_filter_gray.jpg')
plt.close(fig)

# A 9 x 9 Gaussian blur was able to retain more detail than a 9 x 9 mean filter and it was able to remove some noise.
#  The Gaussian filter did not create artifacts for the grayscale image.

#  --------------------
#  MEDIAN FILTER
#  --------------------

new_image = cv2.medianBlur(image, figure_size)

fig = plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)), plt.title('Median Filter')
plt.xticks([]), plt.yticks([])
#  plt.show()
plt.savefig('Output/median_filter.jpg')
plt.close(fig)
# The median filter is able to retain the edges of the image while removing salt-and-pepper noise. Unlike the mean
# and Gaussian filter, the median filter does not produce artifacts on a color image.

# gray
new_image = cv2.medianBlur(image2, figure_size)

plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(image2, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image, cmap='gray'), plt.title('Median Filter')
plt.xticks([]), plt.yticks([])
#  plt.show()
plt.savefig('Output/median_filter_gray.jpg')
plt.close(fig)


#  ------------------------------------------------------
#  CONSERVATIVE SMOOTHENING
#  ----------------------------------------------------
# first a conservative filter for grayscale images will be defined.
def conservative_smoothing_gray(data, filter_size):
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k, m])
            temp.remove(data[i, j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i, j] > max_value:
                new_image[i, j] = max_value
            elif data[i, j] < min_value:
                new_image[i, j] = min_value
            temp = []
    return new_image.copy()


#  Then a 9 x 9 conservative smoothing filter will be applied to a grayscale image.

new_image = conservative_smoothing_gray(image2, 9)

plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(image2, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_image, cmap='gray'), plt.title('Conservative Smoothing')
plt.xticks([]), plt.yticks([])
# plt.show()
plt.savefig('Output/Conservative_smoothing_grey.jpg')
plt.close(fig)

# The conservative smoothing filter was able to retain more detail than the median filter however it was not able to
# remove as much salt-and-pepper noise.

# ----------------------------------
# LAPLACIAN
# ----------------------------------
new_image = cv2.Laplacian(image2, cv2.CV_64F)

plt.figure(figsize=(11, 6))
plt.subplot(131), plt.imshow(image2, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(new_image, cmap='gray'), plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(image2 + new_image, cmap='gray'), plt.title('Resulting image')
plt.xticks([]), plt.yticks([])
# plt.show()
plt.savefig('Output/Laplacian_grey.jpg')
plt.close(fig)

# First the Laplacian of the image will be determined. Then the edges will be enhanced by adding the Laplacian to the
# original image. While the edges may have been enhanced, some of the noise was also enhanced.

# -----------------------------
# FREQUENCY FILTER HPF & LPF
# _____________________________
dft = cv2.dft(np.float32(image2), flags=cv2.DFT_COMPLEX_OUTPUT)

# shift the zero-frequncy component to the center of the spectrum
dft_shift = np.fft.fftshift(dft)

# save image of the image in the fourier domain.
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# plot both images
plt.figure(figsize=(11, 6))
plt.subplot(131), plt.imshow(image2, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


rows, cols = image2.shape
crow, ccol = rows // 2, cols // 2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# apply mask and inverse DFT
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# plot the filtered image
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])
# plt.show()
plt.savefig('Output/LPF_grey.jpg')
