"""
Crea palabras automáticamente detectando señas sostenidas.
Abre la cámara, detecta la mano y forma una palabra a partir de las detecciones.
"""

import os
import cv2
import threading
import pyttsx3
import collections
import mediapipe as mp
import numpy as np
import joblib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ARCHIVO_MODELO = "modelo.pkl"

# Configuración de los umbrales (frames)
FRAMES_PARA_LETRAS = 20  # ~1 segundo a 20-30 FPS
FRAMES_PARA_ESPACIO = 40  # ~1.5 - 2 segundos sin manos para registrar un espacio


def hablar_texto(texto: str):
    """Reproduce el texto de forma asincrona para no congelar la camara."""
    if not texto.strip():
        return
    def tts():
        engine = pyttsx3.init()
        # Ajustar velocidad si se desea
        engine.setProperty("rate", 140)
        engine.say(texto)
        engine.runAndWait()
    
    hilo = threading.Thread(target=tts, daemon=True)
    hilo.start()

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


def detectar_gesto_dinamico(historial_muneca, prediccion_estatica):
    """
    Analiza la trayectoria de la muñeca (Landmark 0) en los últimos frames
    para detectar letras con movimiento (J, Z, X).
    """
    if len(historial_muneca) < 15:
        return prediccion_estatica
        
    y_coords = [p[1] for p in historial_muneca]
    x_coords = [p[0] for p in historial_muneca]
    
    desplazamiento_y = y_coords[-1] - y_coords[0]
    desplazamiento_x = x_coords[-1] - x_coords[0]
    distancia_total = (desplazamiento_x**2 + desplazamiento_y**2)**0.5
    
    max_x = max(x_coords)
    min_x = min(x_coords)

    # Heurística para la J: Pose base 'I' (meñique levantado) + curva hacia abajo
    if prediccion_estatica in ['I', 'J']:
        if desplazamiento_y > 0.08 and distancia_total > 0.1:
            return 'J'
        return 'I'
        
    # Heurística para la Z: Pose base 'D' (índice levantado) + movimiento amplio (zigzag)
    if prediccion_estatica in ['D', 'Z']:
        if distancia_total > 0.15:
            return 'Z'
        return 'D'
        
    # Heurística para la X: Pose base 'X'/'Q' (gancho índice) + tirón hacia abajo
    if prediccion_estatica in ['X', 'Q']:
        if desplazamiento_y > 0.05:
            return 'X'
            
    # Heurística para la Ñ: Pose base 'N' + movimiento horizontal (oscilación)
    if prediccion_estatica in ['N', 'Ñ']:
        if (max_x - min_x) > 0.06:
            return 'Ñ'
        return 'N'
            
    return prediccion_estatica


def main():
    if not os.path.exists(ARCHIVO_MODELO):
        print(f"Error: No existe {ARCHIVO_MODELO}")
        print("Ejecuta primero: python entrenar_modelo.py")
        return

    modelo = joblib.load(ARCHIVO_MODELO)
    print("Modelo cargado. Modo de Creación de Palabras Automático iniciado.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Variables de estado para la construcción de palabras
    palabra_actual = ""
    prediccion_actual = None
    conteo_frames_letra = 0
    ultima_letra_agregada = None
    conteo_frames_sin_mano = 0
    
    # Buffer para el historial de la muñeca (últimos 20 frames)
    historial_muneca = collections.deque(maxlen=20)

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
                # Reiniciar el contador de espacio ya que hay una mano visible
                conteo_frames_sin_mano = 0
                
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
                
                # Guardar la coordenada normalizada de la muñeca (x, y)
                historial_muneca.append((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y))
                
                # APLICAR HEURÍSTICA DE MOVIMIENTO
                etiqueta = detectar_gesto_dinamico(historial_muneca, etiqueta)

                if confianza > 0.6:  # Solo confiar si hay buena probabilidad
                    # Si la etiqueta es la misma que estamos viendo, incrementamos contador
                    if etiqueta == prediccion_actual:
                        conteo_frames_letra += 1
                    else:
                        prediccion_actual = etiqueta
                        conteo_frames_letra = 1
                    
                    # Logica para agregar una letra a la palabra
                    if conteo_frames_letra >= FRAMES_PARA_LETRAS:
                        if etiqueta != ultima_letra_agregada:
                            palabra_actual += etiqueta
                            ultima_letra_agregada = etiqueta
                            conteo_frames_letra = 0  # Reiniciar despues de agregar

                # UI: Mostrar Progreso para registrar Letra
                cv2.putText(
                    frame, f"Sena detectada: {etiqueta} ({confianza:.0%})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                
                progreso = min(conteo_frames_letra / FRAMES_PARA_LETRAS, 1.0)
                if progreso > 0 and etiqueta != ultima_letra_agregada:
                    cv2.rectangle(frame, (10, 60), (10 + int(200 * progreso), 80), (0, 255, 255), -1)
                    cv2.rectangle(frame, (10, 60), (210, 80), (255, 255, 255), 2)
                elif etiqueta == ultima_letra_agregada:
                    # Mostrar que la letra ya fue capturada y espera a que retires la mano o cambies de seña
                    cv2.rectangle(frame, (10, 60), (210, 80), (0, 255, 0), -1)
                    cv2.putText(frame, "Capturada", (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            else:
                conteo_frames_sin_mano += 1
                prediccion_actual = None
                conteo_frames_letra = 0
                historial_muneca.clear()
                
                # Resetea la ultima letra cuando la mano desaparece unos instantes, 
                # así puedes tipear la misma letra dos veces separando la mano.
                if conteo_frames_sin_mano > 10:
                    ultima_letra_agregada = None
                
                # Si pasa suficiente tiempo sin mano, agregamos un espacio y paramos de contar
                if conteo_frames_sin_mano == FRAMES_PARA_ESPACIO:
                    if len(palabra_actual) > 0 and palabra_actual[-1] != " ":
                        palabra_actual += " "
                
                cv2.putText(
                    frame, "Coloca la mano frente a la camara",
                    (10, alto // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
                
                # UI: Mostrar Progreso para registrar Espacio
                if conteo_frames_sin_mano < FRAMES_PARA_ESPACIO and len(palabra_actual) > 0 and palabra_actual[-1] != " ":
                    progreso = conteo_frames_sin_mano / FRAMES_PARA_ESPACIO
                    cv2.putText(frame, "Agregando espacio...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.rectangle(frame, (10, 70), (10 + int(200 * progreso), 80), (255, 255, 0), -1)
                    cv2.rectangle(frame, (10, 70), (210, 80), (255, 255, 255), 1)

            # UI: Mostrar la palabra construida en la parte inferior
            cv2.rectangle(frame, (0, alto - 80), (ancho, alto), (0, 0, 0), -1)
            cv2.putText(
                frame, f"Palabra: {palabra_actual}",
                (20, alto - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
            )
            
            # Instrucciones en texto pequeño
            cv2.putText(
                frame, "L: Limpiar | BORRAR: Deshacer | V: Hablar | Q: Salir",
                (ancho - 480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )

            cv2.imshow("Crear Palabras - Modo Automatico", frame)
            
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("l"):
                palabra_actual = ""
            elif tecla == ord("v"):
                hablar_texto(palabra_actual)
            elif tecla == 8 or tecla == 127: # Tecla Backspace
                if len(palabra_actual) > 0:
                    palabra_actual = palabra_actual[:-1]
                # Reiniciar estado
                ultima_letra_agregada = None
                prediccion_actual = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
