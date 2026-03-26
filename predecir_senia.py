"""
Predice la seña en tiempo real usando el modelo entrenado.
Abre la cámara, detecta la mano y muestra la etiqueta predicha.
"""

import os

import cv2
import joblib
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ARCHIVO_MODELO = "modelo.pkl"


def extraer_landmarks(hand_landmarks) -> np.ndarray:
    """Extrae los 21 puntos (x, y, z) normalizados por posicion y escala."""
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
    return puntos_np.reshape(1, -1)


def main():
    if not os.path.exists(ARCHIVO_MODELO):
        print(f"Error: No existe {ARCHIVO_MODELO}")
        print("Ejecuta primero: python entrenar_modelo.py")
        return

    modelo = joblib.load(ARCHIVO_MODELO)
    print("Modelo cargado. Presiona Q para salir.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    ) as hands:
        while cap.isOpened():
            exito, frame = cap.read()
            if not exito:
                break

            frame = cv2.flip(frame, 1)
            alto, ancho, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(rgb)

            if resultados.multi_hand_landmarks:
                hand_landmarks = resultados.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                puntos = extraer_landmarks(hand_landmarks)
                etiqueta = modelo.predict(puntos)[0]
                proba = modelo.predict_proba(puntos)[0]
                confianza = max(proba)

                cv2.putText(
                    frame, f"Sena: {etiqueta} ({confianza:.0%})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
                )
            else:
                cv2.putText(
                    frame, "Coloca la mano frente a la camara",
                    (10, alto // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow("Predecir Sena", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
