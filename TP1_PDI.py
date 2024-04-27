import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carga de imágenes en escala de grises
examen_1 = cv2.imread('multiple_choice_1.png', cv2.IMREAD_GRAYSCALE)
examen_2 = cv2.imread('multiple_choice_2.png', cv2.IMREAD_GRAYSCALE)
examen_3 = cv2.imread('multiple_choice_3.png', cv2.IMREAD_GRAYSCALE)
examen_4 = cv2.imread('multiple_choice_4.png', cv2.IMREAD_GRAYSCALE)
examen_5 = cv2.imread('multiple_choice_5.png', cv2.IMREAD_GRAYSCALE)

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)





def campos_ecabezado_y_multiple_choise(imagen_bool):
    # Sumamos los valores en cada fila para encontrar el contenido de cada una
    img_rows = np.sum(imagen_bool, 1)

    # Encontramos las filas que superan un cierto umbral de contenido
    filas_validas = img_rows > 500

    # Obtenemos los índices de todas las filas válidas
    indices_validos = np.where(filas_validas)[0]
    
    # Sabemos que el recuadro está entre un espacio de 20
    umbral_espacio = 22  # Ajustar el umbral según la disposición específica de los datos

    for i in range(1, len(indices_validos)):
        if indices_validos[i] - indices_validos[i - 1] > umbral_espacio:
            primer_espacio_index = i
            break

    # Índice del final de la sección antes del espacio de interés
    indice_inicial = indices_validos[primer_espacio_index - 1] + 1
    # Índice del comienzo de la sección después del espacio de interés
    indice_final = indices_validos[primer_espacio_index] - 1

    # Recortamos la imagen para incluir solo la sección entre estos índices
    img_final = imagen_bool[indice_inicial:indice_final]

    '''
    #Mostramos la imagen recortada de los campos
    plt.imshow(img_final, cmap='gray')
    plt.show()
    '''

    # Sumamos los valores en cada columna para encontrar el contenido de cada una
    img_cols = np.sum(img_final, 0)

    # Identificamos columnas que tienen un contenido significativo. Definimos 19 pix
    umbral_columna_valida = 19
    columnas_validas = img_cols > umbral_columna_valida

    # Definimos listas para guardar los índices de inicio y fin de las áreas sin columnas válidas
    inicio_sub_imgs = []
    fin_sub_imgs = []

    # Identificamos las áreas continuas de columnas no válidas
    i = 0
    while i < len(columnas_validas):
        if not columnas_validas[i]:
            inicio = i
            while i < len(columnas_validas) and not columnas_validas[i]:
                i += 1
            fin = i - 1
            inicio_sub_imgs.append(inicio)
            fin_sub_imgs.append(fin)
        i += 1
    
    nombre = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[2]:fin_sub_imgs[2]]
    id = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[4]:fin_sub_imgs[4]]
    tipo = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[6]:fin_sub_imgs[6]]
    fecha = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[8]:fin_sub_imgs[8]]
    multi_choise = imagen_bool[indice_final + 2:, :]


    '''
    # Mostramos la imagen recortada del nombre
    plt.imshow(nombre, cmap='gray')
    plt.show()
    # Mostramos la imagen recortada del ID
    plt.imshow(id, cmap='gray')
    plt.show()
    # Mostramos la imagen recortada del tipo
    plt.imshow(tipo, cmap='gray')
    plt.show()
    # Mostramos la imagen recortada de la fecha
    plt.imshow(fecha, cmap='gray')
    plt.show()
    # Mostramos la imagen recortada del multiple choise
    plt.imshow(multi_choise, cmap='gray')
    plt.show()
    '''

    return nombre, id, tipo, fecha, multi_choise



#Obtenemos cantidad de palabras y caracteres
def comp_conectados_espacios(img):    
    f_point = img
    f_point = f_point.astype(np.uint8)
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(f_point, connectivity, cv2.CV_32S)
    caracteres = 0
    palabras = 0
    if  num_labels <= 2:
        caracteres = num_labels - 1
        palabras = num_labels - 1
    else:   
        ind_ord = np.argsort(stats[:,0])
        stats_ord = stats[ind_ord]
        resultados = []
        #print(stats) 
        for i in range(2,num_labels):
            fila_actual = stats_ord[i]
            fila_anterior = stats_ord[i - 1]
            suma = fila_actual[0] - (fila_anterior[0] + fila_anterior[2])        
            resultados.append(suma)    
        espacios = 0
        #print(resultados)
        for valor in resultados:
            #Este valor de 5 lo encontré a partir de imprimer resultdos.
            #siginfica que hay un espacio. Esto se calcula con el for anteror
            #a partir de la matriz stats 
            if valor >= 5:
                espacios += 1
        palabras = espacios + 1
        caracteres = num_labels + espacios - 1
    return caracteres, palabras



def validacion(examen, nro):  
    
    def name(nombre):
        caracteres, palabras = comp_conectados_espacios(nombre)
        result_name = "OK" if 1 < caracteres <= 25 and palabras > 1 else "MAL"
        return result_name

    def id(legajo):
        caracteres, palabras = comp_conectados_espacios(legajo)
        result_id = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return result_id

    def code(tipo):
        caracteres, palabras = comp_conectados_espacios(tipo)
        result_code = "OK" if caracteres == 1 else "MAL"
        return result_code 

    def date(dia):
        caracteres, palabras = comp_conectados_espacios(dia)
        result_date = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return result_date

    img_th = examen < 200
    
    nombre, legajo, tipo, dia, multi_choise = campos_ecabezado_y_multiple_choise(img_th)

    '''

    Acá en la variable "nombre" tenemos la imagen del nombre por cada alumno para general la imágen final 
    que pide el ejercicio con el crop de los campos Name en el punto d). Está ubralada, en negativo, habría que pasarla a positivo. 
    Enconté para esto la siguiente función de numpy "Name = np.invert(nombre)". Además, la función 
    campos_ecabezado_y_multiple_choise() devuelve en la variable "multi_choise" la imagen ubralada en bool solamente de
    las preguntas, es decir, se le saca el encabezado y los campos de Name, id, etc., para trabajar directamente sobre las preguntas
    de modo más simple.

    '''

    resultados = {
        "Nombre y Apellido": name(nombre),
        "Legajo": id(legajo),
        "Código": code(tipo),
        "Fecha": date(dia)
    }

    print("+------------------------+-----------+")
    print(f"|         Exámen {nro}              |")
    print("+------------------------+-----------+")
    print("| Campo                  | Resultado |")
    print("+------------------------+-----------+")
    
    for campo, resultado in resultados.items():
        print(f"| {campo:<22} |   {resultado:<7} |")
    print("+------------------------+-----------+")



validacion(examen_1,1)

'''
validacion(examen_2,2)
validacion(examen_3,3)
validacion(examen_4,4)
validacion(examen_5,5)
'''

#Buscamos la salida de la funcion anterior
a,b,c,d,multiple_choice = campos_ecabezado_y_multiple_choise(examen_1)
multiple_choice = multiple_choice < 200

def correccion_examen(multiple_choice_binary):
    respuesta = []
    pregunta = []
    img_rows = np.sum(multiple_choice_binary, 1)

        # Identificamos columnas que tienen un contenido significativo. Definimos 1
    umbral_columna_valida = 20
    columnas_validas = img_rows >= umbral_columna_valida
    x = np.diff(columnas_validas)    
    renglones_indxs = np.argwhere(x)    # Esta variable contendrá todos los inicios y finales de los renglones
    renglones_indxs = renglones_indxs[1:]
    renglones_indxs = renglones_indxs.reshape((-1,2))     # De esta manera, cada fila de letras_indxs contiene el inicio y final de cada letra.

    # Recorro cada par de indices para obtener el renglon de cada respuesta de la 1 a 25
    for i in range(len(renglones_indxs)):
        renglon_val = multiple_choice_binary[renglones_indxs[i][0]:renglones_indxs[i][1], 0:1000]    
        
        # Convierto a binary para Hugues Tranform
        binary_image = renglon_val.astype(np.uint8) * 255

        # Aplicar la transformada de Hough circular (habiendo realizado una busqueda previa de los parametros)
        circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, dp=2, minDist=20, param1=50, param2=20, minRadius=10, maxRadius=20)

        #ordenarlos por su valor en el eje X antes de revisar si estan
        circles = sorted(circles[0, :], key=lambda circle: circle[0])
        
        #Redondeo sus coordenadas para el for  
        circles = np.uint16(np.around(circles))

        #Los enumero con una 4ta dim    
        circles = [np.append(circle, x + 1) for x, circle in enumerate(circles)]    
        
        circles = np.array(circles)
        
        # Crear una copia de la imagen original para dibujar los círculos
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        
        neighborhood_size = 10
                
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Dibujar el círculo en verde
            cv2.circle(output_image, center, radius, (0, 255, 0), 2)
            # Verificar si al menos una cantidad mínima de píxeles dentro del círculo están rellenos de blanco (255)
            x, y = circle[0], circle[1]
            neighborhood_x = slice(max(0, x - neighborhood_size), min(binary_image.shape[1], x + neighborhood_size + 1))
            neighborhood_y = slice(max(0, y - neighborhood_size), min(binary_image.shape[0], y + neighborhood_size + 1))
            neighborhood_pixels = binary_image[neighborhood_y, neighborhood_x]
            is_filled = np.count_nonzero(neighborhood_pixels == 255) >= neighborhood_pixels.size * 0.5  # Al menos el 50% de la vecindad debe estar rellena
            # Si el círculo está relleno de blanco, dibujar un contorno rojo
            if is_filled:
                pregunta = np.append(pregunta, i+1)
                respuesta = np.append(respuesta, circle[3])

                cv2.circle(output_image, center, radius, (0, 0, 255), 2)

        # # Mostrar la imagen con los círculos encontrados
        cv2.imshow('Circles Detected', output_image)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()
    return pregunta, respuesta

    
pregunta, respuesta = correccion_examen(multiple_choice)
imshow(multiple_choice)
len(pregunta)
len(respuesta)