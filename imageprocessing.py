# -*- coding: utf-8 -*-
"""ImageProcessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18nXpitAk_yh7wPcV4p1ssoIPBTn4vrZZ
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
image_path = 'lily.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Periksa apakah gambar dimuat dengan benar
if image is None:
    raise ValueError(f"Image at path '{image_path}' could not be loaded.")

# Konversi tipe data gambar jika diperlukan
if image.dtype != np.uint8:
    image = image.astype(np.uint8)

# Menerapkan algoritma Canny untuk deteksi tepi
edges = cv2.Canny(image, 100, 200)

# Menampilkan gambar asli dan gambar hasil deteksi tepi
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Edges Detected')
plt.imshow(edges, cmap='gray')
plt.show()