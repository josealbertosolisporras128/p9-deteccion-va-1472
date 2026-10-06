# ejemplo 3: Dibujar y contar objetos mediante contornos
# jose solis nc 1472

import os
import cv2

# Obtener la ruta de la carpeta donde se encuentra este script ('codigo')
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Construir rutas absolutas automáticas hacia imagenes y resultados
RUTA_IMAGEN = os.path.join(DIR_SCRIPT, "..", "imagenes", "loro.jpg")
RUTA_SALIDA = os.path.join(DIR_SCRIPT, "..", "resultados", "ejemplo3_objetos_1472.jpg")

# 1. Cargar la imagen del loro
imagen = cv2.imread(RUTA_IMAGEN)

# Comprobar si la imagen se cargó correctamente
if imagen is None:
    print("--------------------------------------------------")
    print("ERROR: No se pudo cargar la imagen 'loro.jpg'.")
    print(f"Ruta intentada: {os.path.abspath(RUTA_IMAGEN)}")
    print("--------------------------------------------------")
    exit()

# 2. Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 3. Aplicar desenfoque para reducir el ruido de las plumas y fondo
desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)

# 4. Umbralizado binarizado
_, binaria = cv2.threshold(
    desenfocada,
    127,
    255,
    cv2.THRESH_BINARY
)

# 5. Encontrar contornos externos
contornos, _ = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Copia de la imagen original para dibujar los resultados
resultado = imagen.copy()

# Contador de objetos/regiones detectadas
cantidad = 0

# 6. Analizar cada contorno detectado
for contorno in contornos:
    area = cv2.contourArea(contorno)

    # Aumentamos el área mínima a 1000 px para ignorar pequeñas plumas/ruido
    if area > 1000:
        cantidad += 1

        # Dibujar el contorno exacto en verde
        cv2.drawContours(
            resultado,
            [contorno],
            -1,
            (0, 255, 0),
            2
        )

        # Obtener las coordenadas del rectángulo delimitador
        x, y, ancho, alto = cv2.boundingRect(contorno)

        # Dibujar el rectángulo delimitador en azul
        cv2.rectangle(
            resultado,
            (x, y),
            (x + ancho, y + alto),
            (255, 0, 0),
            2
        )

        # Escribir la etiqueta sobre la caja en rojo
        cv2.putText(
            resultado,
            f"Objeto {cantidad}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

# 7. Mostrar resultados
cv2.imshow("Imagen binaria", binaria)
cv2.imshow("Objetos identificados en Loro", resultado)

# 8. Guardar resultado
os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
cv2.imwrite(RUTA_SALIDA, resultado)

print("Regiones/Objetos principales identificados:", cantidad)
print(f"Resultado guardado en: {os.path.abspath(RUTA_SALIDA)}")

# Esperar tecla y cerrar ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()