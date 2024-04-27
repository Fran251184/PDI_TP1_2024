import cv2
import numpy as np
import matplotlib.pyplot as plt

#Investigamos una función destinada a este propósito llamada CLAHE,
#la cual implementa también un kernel al que le podemos indicar su tamaño (tile) y el contraste que
#usará como guía.

# Umbral para limitar el contraste.
img = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)
clip = 30
tile = 20
clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
equalized = clahe.apply(img)
plt.imshow(equalized,cmap='gray',vmin=0,vmax=255)

plt.show()
