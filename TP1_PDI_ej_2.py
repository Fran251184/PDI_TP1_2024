import cv2 
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt 

# Carga de imágenes en escala de grises
examen_1 = cv2.imread('multiple_choice_1.png', cv2.IMREAD_GRAYSCALE)
examen_2 = cv2.imread('multiple_choice_2.png', cv2.IMREAD_GRAYSCALE)
examen_3 = cv2.imread('multiple_choice_3.png', cv2.IMREAD_GRAYSCALE)
examen_4 = cv2.imread('multiple_choice_4.png', cv2.IMREAD_GRAYSCALE)
examen_5 = cv2.imread('multiple_choice_5.png', cv2.IMREAD_GRAYSCALE)

# Diccionario de respuestas correctas del examen
respuestas_correctas_examen = {1: 1, 2: 1, 3: 2, 4: 1, 5: 4, 6: 2, 7: 2, 8: 3, 9: 2, 10: 1, 11: 4, 12: 1, 13: 3, 14: 3, 15: 4, 16: 2, 17: 1, 18: 3, 19: 3, 20: 4, 21: 2, 22: 1, 23: 3, 24: 3, 25: 3}

examenes = [examen_1, examen_2, examen_3, examen_4, examen_5]

# Función para mostrar imágenes
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

# Función para extraer campos de encabezado y opciones de múltiple elección
def campos_ecabezado_y_multiple_choise(imagen_bool):
    # Suma de valores en cada fila para encontrar el contenido de cada una
    img_rows = np.sum(imagen_bool, 1)

    # Filas que superan un cierto umbral de contenido
    filas_validas = img_rows > 500

    # Obtención de índices de todas las filas válidas
    indices_validos = np.where(filas_validas)[0]
    
    umbral_espacio = 22  

    # Bucle para encontrar los índices de los espacios
    for i in range(1, len(indices_validos)):
        if indices_validos[i] - indices_validos[i - 1] > umbral_espacio:
            primer_espacio_index = i
            break

    # Índices inicial y final antes y después del espacio de interés
    indice_inicial = indices_validos[primer_espacio_index - 1] + 1
    indice_final = indices_validos[primer_espacio_index] - 1
    img_final = imagen_bool[indice_inicial:indice_final]

    # Suma de valores en cada columna para encontrar el contenido de cada una
    img_cols = np.sum(img_final, 0)
    umbral_columna_valida = 19
    columnas_validas = img_cols > umbral_columna_valida

    inicio_sub_imgs = []
    fin_sub_imgs = []

    # Identificación de áreas continuas de columnas no válidas
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
    
    # Extracción de campos específicos
    nombre = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[2]:fin_sub_imgs[2]]
    id = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[4]:fin_sub_imgs[4]]
    tipo = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[6]:fin_sub_imgs[6]]
    fecha = imagen_bool[indice_inicial:indice_final, inicio_sub_imgs[8]:fin_sub_imgs[8]]
    multi_choise = imagen_bool[indice_final + 2:, :]

    return nombre, id, tipo, fecha, multi_choise

# Función para obtener cantidad de palabras y caracteres en campos conectados
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
        for i in range(2,num_labels):
            fila_actual = stats_ord[i]
            fila_anterior = stats_ord[i - 1]
            suma = fila_actual[0] - (fila_anterior[0] + fila_anterior[2])        
            resultados.append(suma)    
        espacios = 0
        for valor in resultados:
            if valor >= 5:
                espacios += 1
        palabras = espacios + 1
        caracteres = num_labels + espacios - 1
    return caracteres, palabras

# Función para extraer respuestas del examen
def respuestas_examen(multiple_choice_binary):
    respuesta = []
    pregunta = []
    img_rows = np.sum(multiple_choice_binary, 1)

    umbral_columna_valida = 20
    columnas_validas = img_rows >= umbral_columna_valida
    x = np.diff(columnas_validas)    
    renglones_indxs = np.argwhere(x)    # Esta variable contendrá todos los inicios y finales de los renglones
    renglones_indxs = renglones_indxs[1:]
    renglones_indxs = renglones_indxs.reshape((-1,2))     

    # Recorrido para obtener el renglón de cada respuesta del 1 al 25
    for i in range(len(renglones_indxs)):
        renglon_val = multiple_choice_binary[renglones_indxs[i][0]:renglones_indxs[i][1], 0:1000]    
        
        # Convertir a binario para Hugues Tranform
        binary_image = renglon_val.astype(np.uint8) * 255

        # Aplicación de la transformada de Hough circular
        circles = cv2.HoughCircles(binary_image, cv2.HOUGH_GRADIENT, dp=2, minDist=20, param1=50, param2=20, minRadius=10, maxRadius=20)

        # Ordenar por su valor en el eje X antes de revisar si están
        circles = sorted(circles[0, :], key=lambda circle: circle[0])
        
        # Redondeo de coordenadas para el bucle  
        circles = np.uint16(np.around(circles))

        # Enumeración con una 4ta dimensión    
        circles = [np.append(circle, x + 1) for x, circle in enumerate(circles)]    
        
        circles = np.array(circles)
        
        # Crear una copia de la imagen original para dibujar los círculos
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        
        neighborhood_size = 10

        cuenta = 0 

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
                guarda_pregunta = np.append(pregunta, i+1)
                guarda_respuesta = np.append(respuesta, circle[3])
                cv2.circle(output_image, center, radius, (0, 0, 255), 2)
                cuenta += 1
            
        if cuenta == 0 or cuenta > 1:

            pregunta = np.append(pregunta, i+1)
            respuesta = np.append(respuesta, 0)
        else:

            pregunta = guarda_pregunta
            respuesta = guarda_respuesta

    dic = dict(zip(pregunta, respuesta))

    return dic

# Función para validar campos de encabezado
def validacion_campos(examen):  
    
    # Validación del nombre
    def name(nombre):
        caracteres, palabras = comp_conectados_espacios(nombre)
        result_name = "OK" if 1 < caracteres <= 25 and palabras > 1 else "MAL"
        return result_name

    # Validación del legajo
    def id(legajo):
        caracteres, palabras = comp_conectados_espacios(legajo)
        result_id = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return result_id

    # Validación del tipo
    def code(tipo):
        caracteres, palabras = comp_conectados_espacios(tipo)
        result_code = "OK" if caracteres == 1 else "MAL"
        return result_code 

    # Validación de la fecha
    def date(dia):
        caracteres, palabras = comp_conectados_espacios(dia)
        result_date = "OK" if caracteres == 8 and palabras == 1 else "MAL"
        return result_date

    # Extracción de campos y múltiple elección
    nombre, legajo, tipo, dia, multi_choice = campos_ecabezado_y_multiple_choise(img_th)

    # Diccionario de validación de campos
    dic = {
        "Nombre y Apellido": name(nombre),
        "Legajo": id(legajo),
        "Código": code(tipo),
        "Fecha": date(dia)
    }

    return dic, multi_choice, nombre

# Función para corregir un examen
def correccion_examen(respuestas):
    correcciones = {}  # Diccionario para almacenar correcciones
    contador_ok = 0  

    for clave in respuestas:
        if clave in respuestas_correctas_examen:  
            if respuestas[clave] == respuestas_correctas_examen[clave]:
                correcciones[clave] = "OK"
                contador_ok += 1  
            else:
                correcciones[clave] = "MAL"

    # Verificar si el estudiante aprobó (20 o más respuestas correctas)
    aprobado = contador_ok >= 20

    return correcciones, aprobado

# Inicialización de la variable de resultados
resultados_concatenados = None  

# Bucle para procesar cada examen
for indice, examen in enumerate(examenes):

    # Umbralización de la imagen
    img_th = examen < 200   

    # Validación de campos y extracción de opciones de múltiple elección
    dic_resultados_validacion, imagen_multi_choice, nombre = validacion_campos(img_th)

    # Extracción de respuestas del examen
    dic_respuestas_preguntas = respuestas_examen(imagen_multi_choice)

    # Corrección del examen y determinación de aprobación
    dic_examen_corregido, aprobado = correccion_examen(dic_respuestas_preguntas)

    # Convertir valores booleanos a valores en escala de grises
    if aprobado:
        nombre_resultado = 255 - nombre  
    else:
        nombre_resultado = 255 + nombre

    # Concatenación de resultados
    if resultados_concatenados is None:
        resultados_concatenados = nombre_resultado
    else:
        resultados_concatenados = np.vstack((resultados_concatenados, nombre_resultado))

    # Impresión de resultados
    print("+------------------------+-----------+")
    print(f"|         EXAMEN {indice+1}                   |")
    print("+------------------------+-----------+")
    print("| Validación Campos      | Resultado |")
    print("+------------------------+-----------+")
        
    for campo, resultado in dic_resultados_validacion.items():
        print(f"| {campo:<22} |   {resultado:<7} |")
        print("+------------------------+-----------+")

    print("+------------------------+-----------+")
    print("|           Corrección               |")
    print("+------------------------+-----------+")
    print("| Preguntas              | Resultado |")
    print("+------------------------+-----------+")

    for pregunta, resultado in dic_examen_corregido.items():
        print(f"| {int(pregunta):<22} |   {resultado:<7} |")
        print("+------------------------+-----------+")

    print("")
    input("PRESIONE ENTER PARA EVALUAR EL PRÓXIMO EXAMEN...")
    print("")


# Normalización de los valores para que puedan ser guardados en una imagen
min_val = np.min(resultados_concatenados)
max_val = np.max(resultados_concatenados)
resultados_normalizados = 255 * (resultados_concatenados - min_val) / (max_val - min_val)
resultados_normalizados = resultados_normalizados.astype(np.uint8)

# Guardar la imagen
cv2.imwrite('resultados_examen.png', resultados_normalizados)

print("SE HA CREADO UNA IMAGEN resultados_examen.png QUE CONTIENE LOS ALUMNOS QUE APROBARON Y REPROBARON")
print("Si en al imagen resultados_examen.png el nombre del alumno figura EN NEGRO, significa que ha APROBADO")
print("Si en al imagen resultados_examen.png el nombre del alumno figura EN BLANCO, significa que ha REPROBADO")
print("")

# Se muestra la imagen de alumnos aprobados (nombre en negro) y reprobados (nombre en blanco)

plt.imshow(resultados_normalizados, cmap='gray')
plt.show()
