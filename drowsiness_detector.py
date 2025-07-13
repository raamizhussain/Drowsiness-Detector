import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
import time
import datetime
import os

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

# NEW: Additional counters and variables
drowsy_count = 0
total_blinks = 0
session_start_time = time.time()
drowsy_episodes = []
last_blink_time = time.time()
awake_status = "AWAKE"
alert_level = 0  # 0=normal, 1=warning, 2=danger

# NEW: Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

def start_alarm():
    global alarm_playing, drowsy_count, alert_level
    if not alarm_playing:
        alarm_sound.play(loops=-1)
        alarm_playing = True
        drowsy_count += 1
        alert_level = 2
        # Log drowsiness episode
        log_drowsiness_episode()
        print(f"Alarm started. Total drowsy episodes: {drowsy_count}")

def stop_alarm():
    global alarm_playing, alert_level
    if alarm_playing:
        alarm_sound.stop()
        alarm_playing = False
        alert_level = 0
        print("Alarm stopped.")

# NEW: Log drowsiness episodes
def log_drowsiness_episode():
    global drowsy_episodes
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drowsy_episodes.append(timestamp)
    
    with open("logs/drowsiness_log.txt", "a") as f:
        f.write(f"Drowsiness detected at: {timestamp}\n")

# NEW: Save screenshot when drowsy
def save_drowsy_screenshot(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/drowsy_screenshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

# NEW: Calculate session duration
def get_session_duration():
    return int(time.time() - session_start_time)

# NEW: Draw status info on frame
def draw_status_info(frame, avg_ear):
    global awake_status, alert_level
    
    # Session timer
    duration = get_session_duration()
    minutes = duration // 60
    seconds = duration % 60
    cv2.putText(frame, f"Session: {minutes}m {seconds}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # EAR value
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Drowsy counter
    cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Total blinks
    cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Alert level indicator
    if alert_level == 0:
        status_color = (0, 255, 0)  # Green
        awake_status = "AWAKE"
    elif alert_level == 1:
        status_color = (0, 255, 255)  # Yellow
        awake_status = "WARNING"
    else:
        status_color = (0, 0, 255)  # Red
        awake_status = "DANGER"
    
    cv2.putText(frame, awake_status, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Simple drowsiness meter (bar)
    bar_width = 200
    bar_height = 20
    bar_x, bar_y = 10, 170
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (100, 100, 100), -1)
    
    # Fill bar based on drowsiness level
    if frame_counter > 0:
        fill_width = min(int((frame_counter / EAR_CONSEC_FRAMES) * bar_width), bar_width)
        color = (0, 255, 255) if frame_counter < EAR_CONSEC_FRAMES else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                      color, -1)
    
    cv2.putText(frame, "Drowsiness Level", (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# NEW: Draw colored eye boxes
def draw_eye_boxes(frame, left_eye, right_eye, avg_ear):
    # Choose color based on EAR value
    if avg_ear < EAR_THRESHOLD:
        color = (0, 0, 255)  # Red for drowsy
    else:
        color = (0, 255, 0)  # Green for awake
    
    # Get bounding boxes for eyes
    left_x = min([point[0] for point in left_eye])
    left_y = min([point[1] for point in left_eye])
    left_w = max([point[0] for point in left_eye]) - left_x
    left_h = max([point[1] for point in left_eye]) - left_y
    
    right_x = min([point[0] for point in right_eye])
    right_y = min([point[1] for point in right_eye])
    right_w = max([point[0] for point in right_eye]) - right_x
    right_h = max([point[1] for point in right_eye]) - right_y
    
    # Draw rectangles around eyes
    cv2.rectangle(frame, (left_x-5, left_y-5), (left_x+left_w+5, left_y+left_h+5), color, 2)
    cv2.rectangle(frame, (right_x-5, right_y-5), (right_x+right_w+5, right_y+right_h+5), color, 2)

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

print("Drowsiness Detection System Started")
print("Press 'q' to quit")
print("Press 's' to save current screenshot")
print("Press 'r' to reset counters")

# NEW: Variables for blink detection
previous_ear = 0.3
blink_detected = False

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

        # NEW: Blink detection
        if previous_ear > EAR_THRESHOLD and avg_ear < EAR_THRESHOLD:
            if not blink_detected:
                total_blinks += 1
                blink_detected = True
                last_blink_time = time.time()
        elif avg_ear > EAR_THRESHOLD:
            blink_detected = False
        
        previous_ear = avg_ear

        # Drowsiness logic with enhanced alerts
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            
            # Warning level
            if frame_counter >= EAR_CONSEC_FRAMES // 2:
                alert_level = 1
            
            # Danger level
            if frame_counter >= EAR_CONSEC_FRAMES:
                start_alarm()
                cv2.putText(frame, "DROWSINESS DETECTED!", (150, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # Save screenshot when very drowsy
                if frame_counter == EAR_CONSEC_FRAMES:
                    save_drowsy_screenshot(frame)
        else:
            frame_counter = 0
            alert_level = 0
            stop_alarm()

        # NEW: Draw all the enhanced features
        draw_status_info(frame, avg_ear)
        draw_eye_boxes(frame, left_eye, right_eye, avg_ear)

        # Draw eye landmarks
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # NEW: Display help text
    cv2.putText(frame, "Press 'q':quit, 's':screenshot, 'r':reset", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Enhanced Drowsiness Detector", frame)

    # NEW: Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current screenshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/manual_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    elif key == ord('r'):
        # Reset counters
        drowsy_count = 0
        total_blinks = 0
        session_start_time = time.time()
        print("Counters reset!")

# NEW: Print session summary
print("\n--- SESSION SUMMARY ---")
print(f"Total session duration: {get_session_duration()} seconds")
print(f"Total drowsy episodes: {drowsy_count}")
print(f"Total blinks detected: {total_blinks}")
print(f"Log files saved in 'logs' directory")

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()