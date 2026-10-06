# ejemplo 2: deteccion de contornos
# jose solis nc 1472

import os
import cv2

# Obtener ruta de la carpeta 'codigo'
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Rutas de entrada y salidas
RUTA_IMAGEN = os.path.join(DIR_SCRIPT, "..", "imagenes", "loro.jpg")
DIR_RESULTADOS = os.path.join(DIR_SCRIPT, "..", "resultados")

# Cargar imagen
imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    print(f"Error: no se pudo cargar la imagen desde: {os.path.abspath(RUTA_IMAGEN)}")
    exit()

# Procesamiento
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)
_, binaria = cv2.threshold(desenfocada, 127, 255, cv2.THRESH_BINARY)

contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

resultado = imagen.copy()
cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)

# Mostrar en pantalla
cv2.imshow("Imagen original 1472", imagen)
cv2.imshow("Imagen binaria 1472", binaria)
cv2.imshow("Contornos detectados 1472", resultado)

# --- GUARDAR TODOS LOS RESULTADOS ---
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# 1. Guardar la original
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo2_original_1472.jpg"), imagen)
# 2. Guardar la binaria
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo2_binaria_1472.jpg"), binaria)
# 3. Guardar el resultado final de contornos
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo2_contornos_1472.jpg"), resultado)

print("--- Ejemplo 2 completado ---")
print("Archivos guardados en carpeta resultados:")
print(" - ejemplo2_original_1472.jpg")
print(" - ejemplo2_binaria_1472.jpg")
print(" - ejemplo2_contornos_1472.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("jose solis nc 1472")