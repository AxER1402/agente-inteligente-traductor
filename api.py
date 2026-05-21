from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuración básica
ARCHIVO_DATASET = "dataset.csv"
ARCHIVO_MODELO = "modelo.pkl"

app = FastAPI(title="SignAI API", description="API para SignAI - Predicción y Recolección")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global
modelo = None
is_training = False

def cargar_modelo():
    global modelo
    if os.path.exists(ARCHIVO_MODELO):
        try:
            modelo = joblib.load(ARCHIVO_MODELO)
            print("✅ Modelo cargado correctamente")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")

cargar_modelo()

class HandLandmarks(BaseModel):
    landmarks: list[float]

class CollectRequest(BaseModel):
    label: str
    landmarks: list[float]

@app.get("/")
def home():
    return {
        "status": "online", 
        "model_loaded": modelo is not None,
        "is_training": is_training
    }

@app.post("/predict")
def predict(data: HandLandmarks):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    try:
        puntos = np.array(data.landmarks).reshape(1, -1)
        etiqueta = modelo.predict(puntos)[0]
        probabilidades = modelo.predict_proba(puntos)[0]
        confianza = float(np.max(probabilidades))
        
        return {
            "prediction": etiqueta,
            "confidence": confianza
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collect")
def collect(data: CollectRequest):
    try:
        archivo_existe = os.path.exists(ARCHIVO_DATASET)
        with open(ARCHIVO_DATASET, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not archivo_existe:
                columnas = [f"{c}{i}" for i in range(21) for c in "xyz"] + ["etiqueta"]
                writer.writerow(columnas)
            
            fila = data.landmarks + [data.label.upper()]
            writer.writerow(fila)
        
        return {"status": "success", "label": data.label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DeleteRequest(BaseModel):
    label: str

@app.post("/delete")
def delete_label(data: DeleteRequest):
    try:
        if not os.path.exists(ARCHIVO_DATASET):
            return {"status": "no_dataset", "message": "El dataset no existe todavía."}
        
        df = pd.read_csv(ARCHIVO_DATASET)
        etiqueta_upper = data.label.upper()
        
        if etiqueta_upper not in df["etiqueta"].values:
            return {"status": "no_samples", "message": f"No hay muestras para la etiqueta {etiqueta_upper}"}
        
        df_nuevo = df[df["etiqueta"] != etiqueta_upper]
        df_nuevo.to_csv(ARCHIVO_DATASET, index=False)
        return {"status": "success", "message": f"Muestras de la etiqueta {etiqueta_upper} eliminadas"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    if not os.path.exists(ARCHIVO_DATASET):
        return {"total_samples": 0, "labels": {}}
    
    try:
        df = pd.read_csv(ARCHIVO_DATASET)
        stats = df["etiqueta"].value_counts().to_dict()
        return {
            "total_samples": len(df),
            "labels": {str(k): int(v) for k, v in stats.items()}
        }
    except Exception as e:
        return {"error": str(e)}

def train_model_task():
    global is_training, modelo
    is_training = True
    try:
        if not os.path.exists(ARCHIVO_DATASET):
            return
        
        df = pd.read_csv(ARCHIVO_DATASET)
        if len(df) < 10:
            return

        X = df.drop(columns=["etiqueta"])
        y = df["etiqueta"]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X, y)
        
        joblib.dump(clf, ARCHIVO_MODELO)
        modelo = clf
        print("✅ Entrenamiento completado y modelo actualizado")
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
    finally:
        is_training = False

@app.post("/train")
def train(background_tasks: BackgroundTasks):
    global is_training
    if is_training:
        return {"status": "already_training"}
    
    background_tasks.add_task(train_model_task)
    return {"status": "training_started"}

@app.get("/reference/{label}")
def get_reference_skeleton(label: str):
    try:
        if not os.path.exists(ARCHIVO_DATASET):
            return {"status": "no_dataset", "message": "El dataset no existe todavía."}
        
        df = pd.read_csv(ARCHIVO_DATASET)
        etiqueta_upper = label.upper()
        
        df_label = df[df["etiqueta"] == etiqueta_upper]
        if df_label.empty:
            return {"status": "no_data", "message": f"No hay datos para la etiqueta {etiqueta_upper}"}
        
        # Calcular el promedio de cada columna para esta etiqueta
        columnas_coords = [c for c in df.columns if c != "etiqueta"]
        coords_promedio = df_label[columnas_coords].mean().to_dict()
        
        puntos = []
        for i in range(21):
            puntos.append({
                "x": coords_promedio.get(f"x{i}", 0.0),
                "y": coords_promedio.get(f"y{i}", 0.0),
                "z": coords_promedio.get(f"z{i}", 0.0)
            })
            
        return {"status": "success", "landmarks": puntos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
