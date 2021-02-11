import imageio
from scipy import fftpack
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.fftpack as fp
import numpy as np

im = imageio.imread('lena.jpg', as_gray=True)  # assuming an RGB image
plt.figure(figsize=(10, 10))
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.show()


F1 = fftpack.fft2(im.astype(float))
F2 = fftpack.fftshift(F1)
plt.figure(figsize=(10, 10))
plt.imshow((20*np.log10(0.1 + F2)).astype(int), cmap=plt.cm.gray)
plt.show()


(w, h) = im.shape
half_w, half_h = int(w/2), int(h/2)

# high pass filter
n = 25
F2[half_w-n:half_w+n+1, half_h-n:half_h+n+1] = 0  # select all but the first 50x50 (low) frequencies
plt.figure(figsize=(10, 10))
plt.imshow((20*np.log10(0.1 + F2)).astype(int))
plt.show()

im1 = fp.ifft2(fftpack.ifftshift(F2)).real
plt.figure(figsize=(10, 10))
plt.imshow(im1, cmap='gray')
plt.axis('off')
plt.show()

