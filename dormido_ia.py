import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound
import datetime
import os

# Configurar el historial en el escritorio del usuario
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
HISTORIAL_FILE = os.path.join(desktop_path, "historial_somnolencia.txt")

# Cargar detector facial y predicción de puntos clave
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Función para calcular la relación de aspecto del ojo (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Parámetros
EAR_THRESHOLD = 0.25  # Umbral para detectar somnolencia
CONSEC_FRAMES = 20    # Número de frames consecutivos para activar alerta
counter = 0           # Contador de frames con ojos cerrados
running = False       # Estado del detector

# Función para registrar intentos de sueño
def registrar_historial():
    fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORIAL_FILE, "a") as file:
        file.write(f"Intento de sueño detectado en: {fecha_hora}\n")
    print(f"📌 Intento de sueño registrado: {fecha_hora}")

# Función para activar/desactivar el detector
def toggle_detector():
    global running
    running = not running
    print("🟢 Detector ACTIVADO" if running else "🔴 Detector DESACTIVADO")

# Captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crear un botón en la ventana
    cv2.putText(frame, "Presiona 's' para INICIAR/PARAR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if running:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [36, 37, 38, 39, 40, 41]])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [42, 43, 44, 45, 46, 47]])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CONSEC_FRAMES:
                    print("⚠️ ¡Despierta!")
                    winsound.Beep(1000, 500)
                    registrar_historial()  # Guardar intento en historial
                    counter = 0  # Reiniciar el contador después de la alarma
            else:
                counter = 0  # Reiniciar si los ojos se abren

    cv2.imshow("Detector de Somnolencia", frame)

    # Teclas para controlar el programa
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Activar/Desactivar detector
        toggle_detector()
    elif key == ord('q'):  # Salir del programa
        break

cap.release()
cv2.destroyAllWindows()
