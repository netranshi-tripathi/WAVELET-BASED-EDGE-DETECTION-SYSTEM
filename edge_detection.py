# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 01:53:49 2024

@author: divyanshi
"""

import numpy as np
import pywt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Load the image in grayscale
img = mpimg.imread('rhombus.png')[:,:,0]

coeffs2 = pywt.dwt2(img, 'db3', mode='symmetric')
cA, (cH, cV, cD)  = coeffs2
imgr = pywt.idwt2(coeffs2, 'db3', mode='symmetric')
imgr = np.uint8(imgr)

plt.figure(figsize=(30, 30))

plt.subplot(2, 2, 1)
plt.imshow(cA, cmap=plt.cm.gray)
plt.title('cA: Approximation coeff.', fontsize=30)

plt.subplot(2, 2, 2)
plt.imshow(cH, cmap=plt.cm.gray)
plt.title('cH: Horizontal detailed coeff.', fontsize=30)

plt.subplot(2, 2, 3)
plt.imshow(cV, cmap=plt.cm.gray)
plt.title('cV: Vertical detailed coeff.', fontsize=30)

plt.subplot(2, 2, 4)
plt.imshow(cD, cmap=plt.cm.gray)
plt.title('cD: Diagonal detailed coeff.', fontsize=30)

plt.show()

plt.imshow(imgr, cmap=plt.cm.gray)
plt.title('Reconstructed image', fontsize=30)

plt.show()
