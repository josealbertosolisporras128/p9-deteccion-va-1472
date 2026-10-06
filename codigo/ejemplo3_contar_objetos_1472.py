# ejemplo 3: Dibujar y contar objetos mediante contornos
# jose solis nc 1472

import os
import cv2

# Obtener ruta de la carpeta 'codigo'
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Rutas de entrada y carpeta de salidas
RUTA_IMAGEN = os.path.join(DIR_SCRIPT, "..", "imagenes", "loro.jpg")
DIR_RESULTADOS = os.path.join(DIR_SCRIPT, "..", "resultados")

# 1. Cargar imagen
imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    print(f"Error: no se pudo cargar la imagen desde: {os.path.abspath(RUTA_IMAGEN)}")
    exit()

# 2. Convertir a escala de grises y suavizado
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)

# 3. Imagen binaria
_, binaria = cv2.threshold(desenfocada, 127, 255, cv2.THRESH_BINARY)

# 4. Encontrar contornos
contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

resultado = imagen.copy()
cantidad = 0

# 5. Analizar y contar objetos
for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area > 1000:
        cantidad += 1
        # Dibujar contorno exacto
        cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 2)
        # Dibujar rectángulo delimitador
        x, y, ancho, alto = cv2.boundingRect(contorno)
        cv2.rectangle(resultado, (x, y), (x + ancho, y + alto), (255, 0, 0), 2)
        # Etiqueta de texto
        cv2.putText(resultado, f"Objeto {cantidad}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# 6. Mostrar ventanas (Las 3 en pantalla)
cv2.imshow("Imagen original 1472", imagen)
cv2.imshow("Imagen binaria 1472", binaria)
cv2.imshow("Objetos identificados en Loro 1472", resultado)

# --- 7. GUARDAR LOS 3 RESULTADOS EN CARPETA ---
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# Imagen 1: Original
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo3_original_1472.jpg"), imagen)
# Imagen 2: Binaria
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo3_binaria_1472.jpg"), binaria)
# Imagen 3: Conteo y Rectángulos (Resultado Final)
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo3_objetos_1472.jpg"), resultado)

print("--- Ejemplo 3 completado ---")
print("Objetos identificados:", cantidad)
print("Archivos guardados en carpeta resultados:")
print(" - ejemplo3_original_1472.jpg")
print(" - ejemplo3_binaria_1472.jpg")
print(" - ejemplo3_objetos_1472.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("jose solis nc 1472")