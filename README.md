# Agente Inteligente - Detector de Lenguaje de Señas

Paso 1: **Detección de mano y extracción de landmarks** usando OpenCV, MediaPipe y NumPy.

## Ejecutar el Proyecto
Para poder abrir ventanas de video y usar la cámara web, hay dos maneras de ejecutar este proyecto dependiendo de tu sistema operativo:

### Opción A: macOS y Windows (Recomendado)
Dado que Docker Desktop (en macOS y Windows) corre sobre una máquina virtual que no tiene acceso directo a la cámara web (`/dev/video0`), **la forma correcta de ejecutarlo es de manera nativa** usando un entorno virtual de Python.

1. Abre tu terminal y navega a la carpeta del proyecto.
2. Crea un entorno virtual e instálalo:
```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno virtual (macOS/Linux)
source venv/bin/activate
# (En Windows usa: venv\Scripts\activate)

# Instalar dependencias
pip install -r requirements.txt
```
3. Ejecuta el detector de mano:
```bash
python detector_mano.py
```

### Opción B: Linux (Docker)
En Linux puro, Docker sí puede acceder directamente a los dispositivos USB y de video del host.
```bash
# Construir y levantar
docker compose up --build

# Entrar al bash del contenedor (opcional)
docker compose exec gesture_detector bash
```


### Controles

- **q** – Salir
- **p** – Imprimir los 21 landmarks en consola (mantén una mano frente a la cámara)

## Paso 3–4: Recolectar dataset

Para crear el dataset de señas:

```bash
python recolectar_dataset.py
```

1. Escribe la etiqueta (ej: A, B, HOLA)
2. Coloca la mano frente a la cámara
3. Presiona **S** para guardar los 21 puntos
4. Repite con la misma postura varias veces para cada letra

### Controles

- **S** – Guardar muestra actual
- **L** – Cambiar etiqueta
- **Q** – Salir

### Salida: dataset.csv

Cada fila: 63 valores (21 puntos × 3 coordenadas) + etiqueta.

## Paso 5–6: Entrenar y predecir

**1. Entrenar el modelo:**

```bash
pip install pandas scikit-learn joblib
python entrenar_modelo.py
```

Genera `modelo.pkl` y muestra la exactitud en consola.

**2. Ver la predicción en tiempo real:**

```bash
python predecir_seña.py
```

Abre la cámara y muestra la etiqueta detectada cuando coloques la mano frente a la cámara. Presiona **Q** para salir.

**3. Crear palabras y hablar (interactivo):**

Para usar el modo automático donde el sistema enlaza letras para formar palabras y las dice en voz alta, primero asegúrate de tener la librería de voz instalada:

```bash
pip install pyttsx3
python crear_palabras.py
```

En este modo:
- Deja tu mano quieta 1 segundo para fijar la letra automáticamente.
- Quita la mano de la cámara por 2 segundos para agregar un espacio automáticamente.
- Presiona **V** para que la IA lea en voz alta la palabra/frase formada.
- Presiona **L** para limpiar la pantalla o **Retroceso** para borrar la última letra.

## Los 21 puntos (landmarks) de la mano

MediaPipe detecta 21 puntos clave por mano:

| Índice | Nombre      | Descripción           |
|--------|-------------|------------------------|
| 0      | WRIST       | Muñeca                 |
| 1-3    | PULGAR      | CMC, IP, TIP           |
| 4-7    | INDICE      | MCP, PIP, DIP, TIP     |
| 8-11   | MEDIO       | MCP, PIP, DIP, TIP     |
| 12-15  | ANULAR      | MCP, PIP, DIP, TIP     |
| 16-19  | MENIQUE     | MCP, PIP, DIP, TIP     |

Cada punto tiene coordenadas normalizadas por posicion (relativas a la muneca) y por escala, ideales para entrenar el modelo de ML y garantizar alta precision sin importar la distancia a la camara.

---

## Preguntas Frecuentes y Arquitectura del Proyecto (FAQ)

### 1. ¿Qué lenguaje estás usando principalmente en el proyecto?
Python (específicamente compatible con Python 3.9+).

### 2. ¿Qué librerías o frameworks estás usando para la IA?
- **MediaPipe:** Para la detección y seguimiento de las manos.
- **OpenCV (`cv2`):** Para capturar el video de la cámara en vivo y dibujar la interfaz gráfica.
- **Scikit-Learn:** Para el algoritmo de Machine Learning.
- **NumPy & Pandas:** Para la manipulación rápida de los datos y cálculos matemáticos (normalización).

### 3. ¿Cómo estás detectando la mano?
Utilizando el módulo **MediaPipe Hands** de Google. Este modelo preentrenado analiza el relieve de la mano y extrae 21 *landmarks* (puntos clave) en un espacio tridimensional (X, Y, Z).

### 4. ¿Qué modelo estás usando para reconocer las señas?
Se utiliza un **RandomForestClassifier** (Clasificador de Bosques Aleatorios). Es un algoritmo clásico de Machine Learning muy rápido y eficiente para datos tabulares (como nuestras 63 coordenadas). No es una Red Neuronal Convolucional (CNN) pura de imágenes, ya que nosotros no le pasamos fotos de la mano al modelo, sino una lista de números (las coordenadas extraídas por MediaPipe).

### 5. ¿El modelo lo entrenaste vos o están usando uno ya preentrenado?
El modelo **lo entrena el propio usuario desde cero**. MediaPipe (que extrae la mano) sí es un modelo preentrenado por Google, pero el "cerebro" que sabe qué significa cada seña se entrena localmente utilizando los datos que el usuario captura con el script `recolectar_dataset.py`.

### 6. ¿Cómo es el flujo del sistema?
El flujo (pipeline) es el siguiente:
1. **Cámara (OpenCV):** Captura el fotograma de video actual.
2. **Detección (MediaPipe):** Encuentra la mano en la imagen y extrae las coordenadas de las 21 articulaciones.
3. **Procesamiento Matemático (NumPy):** Las coordenadas se normalizan (se hacen relativas a la muñeca y se ajusta la escala según la distancia a la cámara). Esto genera 63 valores.
4. **Predicción (Scikit-Learn):** Se envían los 63 valores al `modelo.pkl` (Random Forest), el cual devuelve la letra más probable y su nivel de confianza.
5. **Interfaz Visual (OpenCV):** Se dibuja el esqueleto de la mano, la letra detectada, y la palabra formada en la ventana de video.

### 7. ¿El sistema funciona en tiempo real o por imágenes?
El sistema funciona en **tiempo real**, analizando el flujo de video en vivo (frame a frame) a 30 FPS.

### 8. ¿Están usando base de datos?
No se utiliza una base de datos relacional tradicional (como MySQL o PostgreSQL). Se utiliza un archivo plano **`dataset.csv`** que actúa como "base de datos". 
- **Información que guarda (Columnas):** Tiene 63 columnas numéricas (`x0, y0, z0, x1, y1, z1 ...`) que representan las posiciones tridimensionales de cada dedo, y una columna final llamada `etiqueta` que contiene el nombre de la letra o seña correspondiente a esos puntos.

### 9. ¿El sistema tiene interfaz (pantalla) o solo consola?
Tiene una **interfaz gráfica superpuesta en la ventana de video** generada directamente por OpenCV (`cv2.imshow`). En ella se pueden visualizar barras de progreso, cuadros delimitadores, esqueletos de manos, detecciones en tiempo real y el texto de las palabras generadas.

### 10. ¿Dónde corre el sistema?
Es una **aplicación de escritorio local**. Se ejecuta directamente en la terminal de la computadora (Mac, Windows o Linux) utilizando el intérprete de Python instalado en el sistema.

### 11. ¿Qué salida da exactamente el sistema?
Depende del script que ejecutes:
- Si corres `predecir_senia.py`: Salida en tiempo real de la **letra analizada** en ese milisegundo.
- Si corres `crear_palabras.py`: Da como salida **texto acumulado (palabras formadas)**. Cuenta con un sistema automático de "bloqueo" temporal de señas e inserción inteligente de espacios (detectando la ausencia de manos) para ir escribiendo oraciones letra por letra.
