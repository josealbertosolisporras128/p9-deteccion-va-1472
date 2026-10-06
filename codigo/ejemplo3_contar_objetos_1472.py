# ejemplo 3: Dibujar y contar objetos mediante contornos
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
cantidad = 0

for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area > 1000:
        cantidad += 1
        cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 2)
        x, y, ancho, alto = cv2.boundingRect(contorno)
        cv2.rectangle(resultado, (x, y), (x + ancho, y + alto), (255, 0, 0), 2)
        cv2.putText(resultado, f"Objeto {cantidad}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Mostrar ventanas
cv2.imshow("Imagen binaria 1472", binaria)
cv2.imshow("Objetos identificados en Loro 1472", resultado)

# --- GUARDAR TODOS LOS RESULTADOS ---
os.makedirs(DIR_RESULTADOS, exist_ok=True)

# 1. Guardar la binaria del ejemplo 3
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo3_binaria_1472.jpg"), binaria)
# 2. Guardar la imagen final con el conteo y rectángulos
cv2.imwrite(os.path.join(DIR_RESULTADOS, "ejemplo3_objetos_1472.jpg"), resultado)

print("--- Ejemplo 3 completado ---")
print("Archivos guardados en carpeta resultados:")
print(" - ejemplo3_binaria_1472.jpg")
print(" - ejemplo3_objetos_1472.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("jose solis nc 1472")