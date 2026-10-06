# ejemplo 2: deteccion de contornos
# jose solis nc 1472

import os
import cv2

# Obtiene la ruta absoluta de la carpeta 'codigo'
DIR_CODIGO = os.path.dirname(os.path.abspath(__file__))

# Sube un nivel a la raíz del proyecto y entra a 'imagenes'
RUTA_IMAGEN = os.path.abspath(os.path.join(DIR_CODIGO, "..", "imagenes", "loro.jpg"))
RUTA_SALIDA = os.path.abspath(os.path.join(DIR_CODIGO, "..", "resultados", "ejemplo2_contornos_1472.jpg"))

print(f"Buscando imagen en: {RUTA_IMAGEN}")

# Cargar imagen
imagen = cv2.imread(RUTA_IMAGEN)

if imagen is None:
    print("\n--------------------------------------------------")
    print("ERROR: OpenCV no pudo abrir la imagen.")
    print("--------------------------------------------------")
    print("Por favor revisa:")
    print("1. Que el archivo esté dentro de la carpeta 'imagenes'.")
    print("2. Que no se llame 'loro.png', 'loro.jpeg' o 'Loro.jpg' (las mayúsculas cuentan).")
    exit()

# Convertir a escala de grises y aplicar suavizado
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
desenfocada = cv2.GaussianBlur(gris, (5, 5), 0)

# Umbralización
_, binaria = cv2.threshold(desenfocada, 127, 255, cv2.THRESH_BINARY)

# Detectar contornos
contornos, jerarquia = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Dibujar contornos
resultado = imagen.copy()
cv2.drawContours(resultado, contornos, -1, (0, 255, 0), 2)

# Mostrar resultados
cv2.imshow("Imagen original", imagen)
cv2.imshow("Imagen binaria", binaria)
cv2.imshow("Contornos detectados", resultado)

# Crear carpeta resultados si no existe y guardar
os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
cv2.imwrite(RUTA_SALIDA, resultado)

print("\nCantidad de contornos encontrados:", len(contornos))
print(f"Resultado guardado en: {RUTA_SALIDA}")

cv2.waitKey(0)
cv2.destroyAllWindows()