import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound  # Para la alarma en Windows

# Ruta al archivo de puntos faciales
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Cargar detector facial y predicción de puntos clave
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Función para calcular la relación de aspecto del ojo (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Umbral de somnolencia y contador de cuadros
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
counter = 0

# Inicializar cámara
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
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
                print("¡Despierta!")  # Mensaje en consola
                winsound.Beep(1000, 500)  # Sonido de alerta
        else:
            counter = 0

    cv2.imshow("Detector de Somnolencia", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

