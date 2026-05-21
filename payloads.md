# SignAI - Documentación de Implementación y Conexión

Este documento detalla la integración técnica realizada entre el motor de Inteligencia Artificial (Python) y la interfaz de usuario (React).

## 🏗️ Arquitectura del Sistema

El proyecto utiliza una arquitectura de **Cliente-Servidor** para desacoplar la detección de visión artificial de la lógica de negocio de la interfaz.

- **Backend:** FastAPI (Python 3.11+) - Maneja el modelo `RandomForest`, el entrenamiento y el almacenamiento del dataset.
- **Frontend:** Vite + React (TypeScript) - Maneja la captura de cámara, detección de puntos con MediaPipe y la experiencia de usuario.

---

## 📡 Endpoints de la API (Python)

El servidor corre por defecto en `http://localhost:8000`.

### 1. Predicción Real-Time
- **Ruta:** `/predict`
- **Método:** `POST`
- **Payload:**
  ```json
  {
    "landmarks": [x0, y0, z0, ..., x20, y20, z20] 
  }
  ```
  *(63 valores flotantes normalizados)*
- **Respuesta:**
  ```json
  {
    "prediction": "A",
    "confidence": 0.985
  }
  ```

### 2. Recolección de Dataset
- **Ruta:** `/collect`
- **Método:** `POST`
- **Payload:**
  ```json
  {
    "label": "HOLA",
    "landmarks": [63 floats...]
  }
  ```
- **Acción:** Añade una nueva fila al archivo `dataset.csv`.

### 3. Entrenamiento
- **Ruta:** `/train`
- **Método:** `POST`
- **Acción:** Inicia un proceso en segundo plano (Background Task) que re-entrena el modelo `RandomForest` y actualiza el archivo `modelo.pkl`.

---

## 🛠️ Componentes Clave (React)

### `HandCamera.tsx`
- **Responsabilidad:** Captura el stream de video, inicializa el SDK de MediaPipe Hands y extrae los 21 puntos.
- **Normalización:** Implementa la misma lógica que el script de Python:
  1. Traslada todos los puntos restando las coordenadas de la muñeca (punto 0).
  2. Escala los puntos dividiendo por el valor máximo absoluto para asegurar invarianza al tamaño.

### `RealTimeTranslation.tsx`
- **Lógica de Estabilidad:** Implementa un `STABILITY_THRESHOLD = 15`. Una letra solo se añade al buffer de palabras si es detectada de forma consecutiva durante 15 frames, evitando falsos positivos por parpadeos o ruido en la cámara.

---

## 🚀 Instrucciones de Ejecución

### 1. Iniciar Backend (Python)
Desde la carpeta `agente-inteligente-traductor`:
```bash
pip install -r requirements.txt
python api.py
```

### 2. Iniciar Frontend (React)
Desde la carpeta `ai-frontend/Interfaz para SignAI`:
```bash
npm install
npm run dev
```

---

## 📂 Archivos Generados/Modificados

| Archivo | Función |
| :--- | :--- |
| `api.py` | Servidor FastAPI que conecta el modelo `.pkl` con la web. |
| `HandCamera.tsx` | Integración real con webcam y MediaPipe. |
| `RealTimeTranslation.tsx` | Conexión con la API para traducción en vivo. |
| `CollectDataset.tsx` | Conexión con la API para recolección de muestras. |
| `TrainModel.tsx` | Interfaz para disparar el entrenamiento en el servidor. |
| `requirements.txt` | Actualizado con `fastapi` y `uvicorn`. |
