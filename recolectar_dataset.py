"""
Recolector de dataset para lenguaje de señas - Paso 3 y 4
Abre la cámara, detecta la mano y guarda los 21 puntos al presionar S.
Genera dataset.csv con formato: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,etiqueta
"""

import csv
import os

import cv2
import mediapipe as mp
import numpy as np

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ARCHIVO_DATASET = "dataset.csv"


def extraer_landmarks(hand_landmarks) -> np.ndarray:
    """Extrae los 21 puntos (x, y, z) normalizados por posicion (muneca) y escala."""
    puntos = []
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z
    for lm in hand_landmarks.landmark:
        puntos.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    
    # Normalizacion de escala (tamano)
    puntos_np = np.array(puntos, dtype=np.float32)
    max_val = np.max(np.abs(puntos_np))
    if max_val > 0:
        puntos_np = puntos_np / max_val
    return puntos_np


def guardar_muestra(puntos: np.ndarray, etiqueta: str) -> None:
    """Añade una fila al dataset.csv."""
    fila = list(puntos) + [etiqueta]
    archivo_existe = os.path.exists(ARCHIVO_DATASET)

    with open(ARCHIVO_DATASET, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not archivo_existe:
            columnas = [f"{c}{i}" for i in range(21) for c in "xyz"] + ["etiqueta"]
            writer.writerow(columnas)
        writer.writerow(fila)


def main():
    etiqueta = input("Etiqueta a recolectar (ej: A, B, HOLA): ").strip().upper()
    if not etiqueta:
        etiqueta = "X"

    print(f"\nRecolectando muestras para: {etiqueta}")
    print("  S = guardar puntos")
    print("  L = cambiar etiqueta")
    print("  Q = salir\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    muestras_guardadas = 0

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,  # Solo una mano para el dataset
    ) as hands:
        while cap.isOpened():
            exito, frame = cap.read()
            if not exito:
                break

            frame = cv2.flip(frame, 1)
            alto, ancho, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(rgb)

            puntos_actuales = None

            if resultados.multi_hand_landmarks:
                hand_landmarks = resultados.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                puntos_actuales = extraer_landmarks(hand_landmarks)

            # Info en pantalla
            cv2.putText(frame, f"Etiqueta: {etiqueta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Muestras guardadas: {muestras_guardadas}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "S: guardar | L: etiqueta | Q: salir", (10, alto - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not resultados.multi_hand_landmarks:
                cv2.putText(frame, "Coloca la mano en la camara", (10, alto // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Recolectar Dataset", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("l"):
                nueva = input("Nueva etiqueta: ").strip().upper()
                if nueva:
                    etiqueta = nueva
                    print(f"Etiqueta cambiada a: {etiqueta}")
            elif tecla == ord("s"):
                if puntos_actuales is not None:
                    guardar_muestra(puntos_actuales, etiqueta)
                    muestras_guardadas += 1
                    print(f"  Guardado #{muestras_guardadas}: {etiqueta}")
                else:
                    print("  No se detecto mano. Coloca la mano frente a la camara.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nListo. Dataset guardado en {ARCHIVO_DATASET} ({muestras_guardadas} muestras para {etiqueta})")


if __name__ == "__main__":
    main()
