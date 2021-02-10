import cv2
import numpy as np

img_path = '/home/renefiedel/PycharmProjects/Image-Processing/Images/Lenna_(test_image).png'
img = cv2.imread(img_path)[:, :, 0]  # gray-scale image
img = img[:700, :700]  # crop to 700 x 700

r = 50  # how narrower the window is
ham = np.hamming(700)[:, None]  # 1D hamming
ham2d = np.sqrt(np.dot(ham, ham.T)) ** r  # expand to 2D hamming

f = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
f_shifted = np.fft.fftshift(f)
f_complex = f_shifted[:, :, 0] * 1j + f_shifted[:, :, 1]
f_filtered = ham2d * f_complex

f_filtered_shifted = np.fft.fftshift(f_filtered)
inv_img = np.fft.ifft2(f_filtered_shifted)  # inverse F.T.
filtered_img = np.abs(inv_img)
filtered_img -= filtered_img.min()
filtered_img = filtered_img * 255 / filtered_img.max()
filtered_img = filtered_img.astype(np.uint8)
