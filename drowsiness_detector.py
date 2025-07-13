import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
import time

# ----- EAR Calculation Function -----
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EAR threshold & required consecutive frames
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Frame counter and alarm state flags
frame_counter = 0
alarm_playing = False

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

def start_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_sound.play(loops=-1)
        alarm_playing = True
        print("Alarm started.")

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        alarm_sound.stop()
        alarm_playing = False
        print("Alarm stopped.")

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [shape[i] for i in LEFT_EYE]
        right_eye = [shape[i] for i in RIGHT_EYE]

        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Show EAR on screen
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Drowsiness logic
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                start_alarm()
                cv2.putText(frame, "DROWSINESS DETECTED!", (150, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            frame_counter = 0
            stop_alarm()

        # Draw eye landmarks
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()
