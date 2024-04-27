import cv2
import numpy as np
import matplotlib.pyplot as plt

# Esta función aplica la ecualización del histograma local utilizando cv2.equalizeHist
def local_histogram_equalization(image, window_size):
    # Obtengo el tamaño de la imagen
    rows, cols = image.shape

    # Agrego un borde a la imagen para manejar los píxeles en el borde
    border_size = window_size // 2
    bordered_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)

    # Inicializo la imagen de salida (matriz de todos ceros con la dimensión de la imagen original)
    equalized_image = np.zeros_like(image)

    # Recorro la imagen original y aplico ecualización local del histograma en cada ventana
    for i in range(rows):
        for j in range(cols):
            # Defino los límites de la ventana para calcular el histograma local y aplicar la ecualización del histograma
            row_start = i
            row_end = i + window_size
            col_start = j
            col_end = j + window_size

            # Extraigo la región de la imagen
            region = bordered_image[row_start:row_end, col_start:col_end]

            # Aplico ecualización del histograma utilizando cv2.equalizeHist
            equalized_region = cv2.equalizeHist(region)

            # Asigno el valor de intensidad ecualizado del píxel central a la imagen de salida
            equalized_image[i, j] = equalized_region[border_size, border_size]

    return equalized_image

# Cargo la imagen
image = cv2.imread('Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

# Tamaño de la ventana de procesamiento
window_size = 20

# Aplico la ecualización local del histograma
equalized_image = local_histogram_equalization(image, window_size)

# Aplico un filtro gaussiano para difuminar el ruido
blurred_image = cv2.medianBlur(equalized_image, 3) 

# Mostrar la imagen original y la imagen ecualizada y suavizada
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 3, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Imagen Ecualizada')
plt.subplot(1, 3, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Imagen Ecualizada y Suavizada')
plt.show()

#Pruebas

window_size1 = 10
equalized_image1 = local_histogram_equalization(image, window_size1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(equalized_image1, cmap='gray')
plt.title('Imagen Ecualizada')
plt.show()

window_size3 = 30
equalized_image3 = local_histogram_equalization(image, window_size3)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(equalized_image3, cmap='gray')
plt.title('Imagen Ecualizada')
plt.show()

window_size4 = 50
equalized_image4 = local_histogram_equalization(image, window_size4)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.subplot(1, 2, 2)
plt.imshow(equalized_image4, cmap='gray')
plt.title('Imagen Ecualizada')
plt.show()

