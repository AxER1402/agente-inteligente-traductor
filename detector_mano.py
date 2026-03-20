"""
Detección de mano y extracción de landmarks - Paso 1 del Agente de Lenguaje de Señas
Usa OpenCV (cámara), MediaPipe (detección) y NumPy (datos)
"""

import cv2
import mediapipe as mp
import numpy as np

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Los 21 landmarks de MediaPipe en orden (índice 0=WRIST, 1-4=pulgar, 5-8=índice, etc.)
LANDMARKS_NAMES = [
    "WRIST", "PULGAR_CMC", "PULGAR_IP", "PULGAR_TIP", "INDICE_MCP",
    "INDICE_PIP", "INDICE_DIP", "INDICE_TIP", "MEDIO_MCP", "MEDIO_PIP",
    "MEDIO_DIP", "MEDIO_TIP", "ANULAR_MCP", "ANULAR_PIP", "ANULAR_DIP",
    "ANULAR_TIP", "MENIQUE_MCP", "MENIQUE_PIP", "MENIQUE_DIP", "MENIQUE_TIP"
]


def extraer_landmarks(hand_landmarks, ancho: int, alto: int) -> np.ndarray:
    """
    Convierte los landmarks de MediaPipe a un array NumPy de coordenadas (x, y, z).
    
    MediaPipe devuelve coordenadas normalizadas [0, 1].
    Opcionalmente se pueden convertir a píxeles multiplicando por ancho/alto.
    
    Returns:
        np.ndarray de shape (21, 3) con [x, y, z] para cada punto
    """
    puntos = []
    for lm in hand_landmarks.landmark:
        # Opción: coordenadas normalizadas (para entrenar IA más adelante)
        puntos.append([lm.x, lm.y, lm.z])
    
    return np.array(puntos, dtype=np.float32)


def main():
    # Configuración de la cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
    
    # Configuración de MediaPipe
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2  # Podemos detectar ambas manos
    ) as hands:
        
        print("Detección de mano activada. Presiona 'q' para salir.")
        print("Presiona 'p' para imprimir los landmarks en consola.\n")
        
        frame_count = 0
        
        while cap.isOpened():
            exito, frame = cap.read()
            if not exito:
                break
            
            frame = cv2.flip(frame, 1)  # Espejo para uso más natural
            alto, ancho, _ = frame.shape
            
            # MediaPipe espera RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = hands.process(rgb)
            
            if resultados.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(resultados.multi_hand_landmarks):
                    # Dibujar landmarks y conexiones
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extraer puntos como array NumPy
                    puntos = extraer_landmarks(hand_landmarks, ancho, alto)
                    
                    # Mostrar coordenadas del primer punto en pantalla (ejemplo)
                    mano_label = "Izq" if resultados.multi_handedness[idx].classification[0].label == "Left" else "Der"
                    cv2.putText(
                        frame, f"Mano {mano_label} - {len(puntos)} landmarks",
                        (10, 30 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    
                    # Cada 30 frames, opcional: imprimir landmarks (descomenta si quieres)
                    frame_count += 1
                    if frame_count % 60 == 0:
                        # Descomentar para ver los puntos en consola:
                        # print(f"\n--- Mano {mano_label} ---")
                        # print(f"Shape: {puntos.shape}")
                        # print(f"Ejemplo (pulgar): x={puntos[4,0]:.3f}, y={puntos[4,1]:.3f}, z={puntos[4,2]:.3f}")
                        pass
            
            # Instrucciones en pantalla
            cv2.putText(frame, "q: salir | p: imprimir landmarks", (10, alto - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Detector de Mano - Landmarks", frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q'):
                break
            elif tecla == ord('p') and resultados.multi_hand_landmarks is not None:
                # Imprimir landmarks cuando se presiona 'p'
                for idx, hand_landmarks in enumerate(resultados.multi_hand_landmarks):
                    puntos = extraer_landmarks(hand_landmarks, ancho, alto)
                    mano_label = "Izq" if resultados.multi_handedness[idx].classification[0].label == "Left" else "Der"
                    print(f"\n=== Mano {mano_label} ===")
                    print(f"Array shape: {puntos.shape}  (21 puntos x 3 coordenadas)")
                    n_puntos = len(puntos)
                    for i in range(n_puntos):
                        nombre = LANDMARKS_NAMES[i] if i < len(LANDMARKS_NAMES) else f"P{i}"
                        print(f"  {i:2d} {nombre:15s}: x={puntos[i,0]:.4f} y={puntos[i,1]:.4f} z={puntos[i,2]:.4f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCámara cerrada.")


if __name__ == "__main__":
    main()
