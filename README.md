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

Cada punto tiene coordenadas **normalizadas** (x, y, z) en el rango 0-1, ideales para entrenar modelos de ML más adelante.
