import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
import time
import datetime
import os
import pyttsx3
import threading
import numpy as np
from collections import deque
import json

# ----- EAR Calculation Function -----
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ----- MAR Calculation Function -----
def calculate_MAR(mouth):
    """Calculate Mouth Aspect Ratio for yawn detection"""
    # Vertical mouth landmarks
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# Landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

# Thresholds & frame counters
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
EAR_CONSEC_FRAMES = 75
MAR_CONSEC_FRAMES = 15

# State variables
frame_counter = 0
yawn_counter = 0
alarm_playing = False
drowsy_count = 0
yawn_count = 0
total_blinks = 0
session_start_time = time.time()
drowsy_episodes = []
yawn_episodes = []
last_blink_time = time.time()
awake_status = "AWAKE"
alert_level = 0
longest_awake_streak = 0
current_awake_streak = 0
total_drowsy_time = 0
drowsy_start_time = 0
sound_enabled = True
night_mode = False
show_landmarks = True
detection_confidence = 0.0
system_paused = False

# Voice engine
voice_engine = pyttsx3.init()
voice_engine.setProperty('rate', 150)
voice_engine.setProperty('volume', 0.8)

# History tracking
ear_history = deque(maxlen=100)
mar_history = deque(maxlen=100)
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# Achievements
achievements = {
    "first_hour": False,
    "no_drowsy_30min": False,
    "blink_master": False,
    "marathon_driver": False,
    "yawn_detector": False
}

# Sound system
alert_sounds = {}

# Create logs directory
if not os.path.exists("logs"):
    os.makedirs("logs")

# Initialize pygame mixer
pygame.mixer.init()
try:
    if os.path.exists("alarm.wav"):
        alert_sounds["danger"] = pygame.mixer.Sound("alarm.wav")
    if os.path.exists("warning.wav"):
        alert_sounds["warning"] = pygame.mixer.Sound("warning.wav")
    if os.path.exists("yawn.wav"):
        alert_sounds["yawn"] = pygame.mixer.Sound("yawn.wav")
except pygame.error:
    print("Could not load sound files")

def play_voice_alert(message):
    """Play voice alert in separate thread"""
    def speak():
        if sound_enabled:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.8)
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                print(f"Voice error: {e}")
    
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

def start_alarm():
    global alarm_playing, drowsy_count, alert_level, drowsy_start_time
    if not alarm_playing and sound_enabled:
        if "danger" in alert_sounds:
            alert_sounds["danger"].play(loops=-1)
        alarm_playing = True
        drowsy_count += 1
        alert_level = 2
        drowsy_start_time = time.time()
        
        if drowsy_count == 1:
            play_voice_alert("Driver alert! Please take a break")
        elif drowsy_count > 3:
            play_voice_alert("Severe drowsiness detected! Pull over safely")
        
        log_drowsiness_episode()

def stop_alarm():
    global alarm_playing, alert_level, total_drowsy_time, drowsy_start_time
    if alarm_playing:
        pygame.mixer.stop()
        alarm_playing = False
        alert_level = 0
        
        if drowsy_start_time > 0:
            total_drowsy_time += time.time() - drowsy_start_time
            drowsy_start_time = 0

def play_warning_sound():
    if sound_enabled and not alarm_playing:
        if "warning" in alert_sounds:
            alert_sounds["warning"].play()

def detect_yawn():
    global yawn_counter, yawn_count
    if yawn_counter >= MAR_CONSEC_FRAMES:
        yawn_count += 1
        yawn_counter = 0
        
        if "yawn" in alert_sounds and sound_enabled:
            alert_sounds["yawn"].play()
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        yawn_episodes.append(timestamp)
        
        with open("logs/yawn_log.txt", "a") as f:
            f.write(f"Yawn detected at: {timestamp}\n")
        
        if yawn_count == 1:
            play_voice_alert("Yawn detected")
        elif yawn_count >= 3:
            play_voice_alert("Multiple yawns detected. Consider taking a break")

def log_drowsiness_episode():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drowsy_episodes.append(timestamp)
    with open("logs/drowsiness_log.txt", "a") as f:
        f.write(f"Drowsiness detected at: {timestamp}\n")

def save_drowsy_screenshot(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/drowsy_screenshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

def get_session_duration():
    return int(time.time() - session_start_time)

def calculate_fps():
    global fps_counter, fps_start_time, current_fps
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()

def check_achievements():
    global achievements
    duration = get_session_duration()
    
    if duration >= 3600 and not achievements["first_hour"]:
        achievements["first_hour"] = True
        play_voice_alert("Achievement unlocked: One hour of monitoring!")
    
    if current_awake_streak >= 1800 and not achievements["no_drowsy_30min"]:
        achievements["no_drowsy_30min"] = True
        play_voice_alert("Achievement unlocked: Thirty minutes alert!")
    
    if total_blinks > 100 and not achievements["blink_master"]:
        achievements["blink_master"] = True
        play_voice_alert("Achievement unlocked: Blink master!")
    
    if duration >= 7200 and not achievements["marathon_driver"]:
        achievements["marathon_driver"] = True
        play_voice_alert("Achievement unlocked: Marathon driver!")
    
    if yawn_count >= 5 and not achievements["yawn_detector"]:
        achievements["yawn_detector"] = True
        play_voice_alert("Achievement unlocked: Yawn detector!")

def draw_ear_graph(frame):
    if len(ear_history) < 2:
        return
    
    graph_x, graph_y = frame.shape[1] - 250, 50
    graph_w, graph_h = 200, 80
    
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
    
    threshold_y = graph_y + graph_h - int((EAR_THRESHOLD / 0.5) * graph_h)
    cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), (0, 0, 255), 1)
    
    points = []
    for i, ear in enumerate(ear_history):
        x = graph_x + int((i / len(ear_history)) * graph_w)
        y = graph_y + graph_h - int((ear / 0.5) * graph_h)
        y = max(graph_y, min(graph_y + graph_h, y))
        points.append((x, y))
    
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
    
    cv2.putText(frame, "EAR Graph", (graph_x, graph_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_mar_graph(frame):
    if len(mar_history) < 2:
        return
    
    graph_x, graph_y = frame.shape[1] - 250, 150
    graph_w, graph_h = 200, 80
    
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
    
    threshold_y = graph_y + graph_h - int((MAR_THRESHOLD / 1.0) * graph_h)
    cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), (0, 0, 255), 1)
    
    points = []
    for i, mar in enumerate(mar_history):
        x = graph_x + int((i / len(mar_history)) * graph_w)
        y = graph_y + graph_h - int((mar / 1.0) * graph_h)
        y = max(graph_y, min(graph_y + graph_h, y))
        points.append((x, y))
    
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (255, 255, 0), 2)
    
    cv2.putText(frame, "MAR Graph", (graph_x, graph_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_enhanced_status(frame, avg_ear, avg_mar, face_rect):
    global awake_status, alert_level, current_awake_streak, longest_awake_streak
    
    if alert_level == 0:
        current_awake_streak += 1
        if current_awake_streak > longest_awake_streak:
            longest_awake_streak = current_awake_streak
    else:
        current_awake_streak = 0
    
    duration = get_session_duration()
    minutes = duration // 60
    seconds = duration % 60
    blinks_per_minute = (total_blinks / max(duration, 1)) * 60
    drowsy_percentage = (total_drowsy_time / max(duration, 1)) * 100
    
    text_color = (100, 100, 100) if night_mode else (255, 255, 255)
    bg_color = (30, 30, 30) if night_mode else (0, 0, 0)
    
    # overlay = frame.copy()
    # cv2.rectangle(overlay, (0, 0), (400, 300), bg_color, -1)
    # cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Session info
    cv2.putText(frame, f"Session: {minutes}m {seconds}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"EAR: {avg_ear:.3f} | MAR: {avg_mar:.3f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f"FPS: {current_fps}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Statistics
    cv2.putText(frame, f"Drowsy: {drowsy_count} | Yawns: {yawn_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Blinks: {total_blinks} | Rate: {blinks_per_minute:.1f}/min", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Drowsy Time: {total_drowsy_time:.1f}s ({drowsy_percentage:.1f}%)", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Status
    if alert_level == 0:
        status_color = (0, 255, 0)
        awake_status = "AWAKE"
    elif alert_level == 1:
        status_color = (0, 255, 255)
        awake_status = "WARNING"
    else:
        intensity = int(128 + 127 * abs(np.sin(time.time() * 5)))
        status_color = (0, 0, intensity)
        awake_status = "DANGER"
    
    cv2.putText(frame, awake_status, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

def draw_face_detection_box(frame, face_rect):
    if face_rect is not None:
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        corner_length = 20
        thickness = 3
        color = (0, 255, 0) if alert_level == 0 else (0, 0, 255)
        
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)

def save_session_data():
    data = {
        "session_duration": get_session_duration(),
        "drowsy_count": drowsy_count,
        "yawn_count": yawn_count,
        "total_blinks": total_blinks,
        "total_drowsy_time": total_drowsy_time,
        "longest_awake_streak": longest_awake_streak,
        "achievements": achievements,
        "ear_threshold": EAR_THRESHOLD,
        "mar_threshold": MAR_THRESHOLD,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open("logs/session_data.json", "w") as f:
        json.dump(data, f, indent=2)

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

print("🚗 ADVANCED DROWSINESS & YAWN DETECTION SYSTEM 🚗")
print("=" * 50)
print("Controls:")
print("'q' - Quit | 's' - Screenshot | 'r' - Reset counters")
print("'m' - Toggle sound | 'n' - Night mode | 'l' - Landmarks")
print("'p' - Pause/Resume | '+/-' - Adjust sensitivity")
print("=" * 50)

# Main loop variables
previous_ear = 0.3
blink_detected = False
last_break_reminder = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    calculate_fps()
    
    if system_paused:
        cv2.putText(frame, "SYSTEM PAUSED - Press 'p' to resume", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Drowsiness & Yawn Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            system_paused = False
        elif key == ord('q'):
            break
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    face_detected = False
    current_face_rect = None
    avg_ear = 0.0
    avg_mar = 0.0

    for face in faces:
        face_detected = True
        current_face_rect = face
        detection_confidence = 85.0
        
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [shape[i] for i in LEFT_EYE]
        right_eye = [shape[i] for i in RIGHT_EYE]
        mouth = [shape[i] for i in MOUTH]

        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_history.append(avg_ear)
        
        avg_mar = calculate_MAR(mouth)
        mar_history.append(avg_mar)

        # Blink detection
        if previous_ear > EAR_THRESHOLD and avg_ear < EAR_THRESHOLD:
            if not blink_detected:
                total_blinks += 1
                blink_detected = True
                last_blink_time = time.time()
        elif avg_ear > EAR_THRESHOLD:
            blink_detected = False
        
        previous_ear = avg_ear

        # Yawn detection
        if avg_mar > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            if yawn_counter > 0:
                detect_yawn()
            yawn_counter = 0

        # Drowsiness detection
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            
            if frame_counter >= EAR_CONSEC_FRAMES // 4:
                alert_level = 1
                if frame_counter == EAR_CONSEC_FRAMES // 2:
                    play_warning_sound()
            
            if frame_counter >= EAR_CONSEC_FRAMES:
                start_alarm()
                cv2.putText(frame, "DROWSINESS DETECTED!", (150, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                if frame_counter == EAR_CONSEC_FRAMES:
                    save_drowsy_screenshot(frame)
        else:
            frame_counter = 0
            alert_level = 0
            stop_alarm()

        # Draw face and landmarks
        draw_face_detection_box(frame, current_face_rect)
        if show_landmarks:
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in mouth:
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    # Break reminder
    if time.time() - last_break_reminder > 1800:
        play_voice_alert("Break reminder: Consider taking a rest")
        last_break_reminder = time.time()

    check_achievements()
    draw_enhanced_status(frame, avg_ear if face_detected else 0, avg_mar if face_detected else 0, current_face_rect)
    draw_ear_graph(frame)
    draw_mar_graph(frame)

    help_text = "q:quit | s:screenshot | r:reset | m:sound | n:night | l:landmarks | p:pause | +/-:sensitivity"
    cv2.putText(frame, help_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Drowsiness & Yawn Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/manual_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
    elif key == ord('r'):
        drowsy_count = 0
        yawn_count = 0
        total_blinks = 0
        session_start_time = time.time()
        total_drowsy_time = 0
        longest_awake_streak = 0
        current_awake_streak = 0
        ear_history.clear()
        mar_history.clear()
    elif key == ord('m'):
        sound_enabled = not sound_enabled
        if not sound_enabled:
            stop_alarm()
    elif key == ord('n'):
        night_mode = not night_mode
    elif key == ord('l'):
        show_landmarks = not show_landmarks
    elif key == ord('p'):
        system_paused = not system_paused
    elif key == ord('+') or key == ord('='):
        EAR_THRESHOLD = min(0.35, EAR_THRESHOLD + 0.01)
        MAR_THRESHOLD = min(0.8, MAR_THRESHOLD + 0.05)
    elif key == ord('-'):
        EAR_THRESHOLD = max(0.15, EAR_THRESHOLD - 0.01)
        MAR_THRESHOLD = max(0.3, MAR_THRESHOLD - 0.05)

save_session_data()

print("\n" + "="*50)
print("📊 SESSION SUMMARY")
print("="*50)
print(f"⏱️  Duration: {get_session_duration()//60}m {get_session_duration()%60}s")
print(f"😴 Drowsy Episodes: {drowsy_count}")
print(f"🥱 Yawns Detected: {yawn_count}")
print(f"👁️  Total Blinks: {total_blinks}")
print(f"💤 Drowsy Time: {total_drowsy_time:.1f}s")
print(f"🏆 Longest Awake Streak: {longest_awake_streak//30}s")
print(f"🎯 Final Thresholds - EAR: {EAR_THRESHOLD:.3f}, MAR: {MAR_THRESHOLD:.3f}")
print(f"🏅 Achievements Unlocked: {sum(achievements.values())}/5")
print("="*50)

stop_alarm()
cap.release()
cv2.destroyAllWindows()
voice_engine.stop()