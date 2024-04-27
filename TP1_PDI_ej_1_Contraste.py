import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargo la imagen
img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

m = np.mean(img)
E = 0.1

img2 = 1 / (1 + (m/(img + np.finfo(float).eps) )**E)

plt.subplot(121)
h=plt.imshow(img,cmap='gray')
plt.colorbar(h)
plt.title('Imagen Original')
plt.subplot(122)
h=plt.imshow(img2, cmap='gray')
plt.colorbar(h)
plt.title('Estrechado de constraste')
plt.show()