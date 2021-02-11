from scipy import fftpack
import numpy as np
import imageio
from PIL import Image, ImageDraw

image1 = imageio.imread('lena.jpg', as_gray=True)

image1_np = np.array(image1)  # convert image to numpy array

fft1 = fftpack.fftshift(fftpack.fft2(image1_np))  # fft of image

x, y = image1_np.shape[0], image1_np.shape[1]  # Create a low pass filter image

e_x, e_y = 70, 70  # size of circle

bbox = ((x / 2) - (e_x / 2), (y / 2) - (e_y / 2), (x / 2) + (e_x / 2), (y / 2) + (e_y / 2))  # create a box

low_pass = Image.new("L", (image1_np.shape[0], image1_np.shape[1]), color=0)

draw1 = ImageDraw.Draw(low_pass)
draw1.ellipse(bbox, fill=1)

low_pass_np = np.array(low_pass)

filtered = np.multiply(fft1, low_pass_np)  # multiply both the images

# inverse fft
ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(filtered)))
ifft2 = np.maximum(0, np.minimum(ifft2, 255))

# save the image
imageio.imsave('fft-then-ifft.png', ifft2.astype(np.uint8))

#
