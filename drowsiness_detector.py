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

# Eye landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EAR threshold & required consecutive frames
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

# Frame counter and alarm state flags
frame_counter = 0
alarm_playing = False

# Additional counters and variables
drowsy_count = 0
total_blinks = 0
session_start_time = time.time()
drowsy_episodes = []
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

# NEW: Voice alert system
voice_engine = pyttsx3.init()
voice_engine.setProperty('rate', 150)
voice_engine.setProperty('volume', 0.8)

# NEW: EAR history for real-time graph
ear_history = deque(maxlen=100)
fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# NEW: Achievement system
achievements = {
    "first_hour": False,
    "no_drowsy_30min": False,
    "blink_master": False,
    "marathon_driver": False
}

# NEW: Multiple alert sounds
alert_sounds = {}

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Initialize pygame mixer for multiple sounds
pygame.mixer.init()
try:
    # Try to load different sounds, fallback to single alarm if not available
    if os.path.exists("alarm.wav"):
        alert_sounds["danger"] = pygame.mixer.Sound("alarm.wav")
    if os.path.exists("warning.wav"):
        alert_sounds["warning"] = pygame.mixer.Sound("warning.wav")
    if os.path.exists("gentle.wav"):
        alert_sounds["gentle"] = pygame.mixer.Sound("gentle.wav")
    
    # If no separate sounds, use the main alarm for all
    if not alert_sounds and os.path.exists("alarm.wav"):
        alarm_sound = pygame.mixer.Sound("alarm.wav")
        alert_sounds["danger"] = alarm_sound
        alert_sounds["warning"] = alarm_sound
        alert_sounds["gentle"] = alarm_sound
    
    # If no alarm.wav, create a simple beep
    if not alert_sounds:
        print("No alarm.wav found - using system beep")
        
except pygame.error:
    print("Could not load sound files")

def play_voice_alert(message):
    """Play voice alert in separate thread"""
    def speak():
        if sound_enabled:
            voice_engine.say(message)
            voice_engine.runAndWait()
    
    thread = threading.Thread(target=speak)
    thread.daemon = True
    thread.start()

def start_alarm():
    global alarm_playing, drowsy_count, alert_level, drowsy_start_time, total_drowsy_time
    if not alarm_playing and sound_enabled:
        if "danger" in alert_sounds:
            alert_sounds["danger"].play(loops=-1)
        alarm_playing = True
        drowsy_count += 1
        alert_level = 2
        drowsy_start_time = time.time()
        
        # Voice alert
        if drowsy_count == 1:
            play_voice_alert("Driver alert! Please take a break")
        elif drowsy_count > 3:
            play_voice_alert("Severe drowsiness detected! Pull over safely")
        
        # Log drowsiness episode
        log_drowsiness_episode()
        print(f"Alarm started. Total drowsy episodes: {drowsy_count}")

def stop_alarm():
    global alarm_playing, alert_level, total_drowsy_time, drowsy_start_time
    if alarm_playing:
        pygame.mixer.stop()
        alarm_playing = False
        alert_level = 0
        
        # Add to total drowsy time
        if drowsy_start_time > 0:
            total_drowsy_time += time.time() - drowsy_start_time
            drowsy_start_time = 0
        
        print("Alarm stopped.")

def play_warning_sound():
    """Play gentler warning sound"""
    if sound_enabled and not alarm_playing:
        if "warning" in alert_sounds:
            alert_sounds["warning"].play()
        elif "gentle" in alert_sounds:
            alert_sounds["gentle"].play()

def log_drowsiness_episode():
    global drowsy_episodes
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drowsy_episodes.append(timestamp)
    
    with open("logs/drowsiness_log.txt", "a") as f:
        f.write(f"Drowsiness detected at: {timestamp}\n")

def save_drowsy_screenshot(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/drowsy_screenshot_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

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
    """Check and unlock achievements"""
    global achievements
    duration = get_session_duration()
    
    # First hour achievement
    if duration >= 3600 and not achievements["first_hour"]:
        achievements["first_hour"] = True
        play_voice_alert("Achievement unlocked: One hour of monitoring!")
    
    # No drowsiness for 30 minutes
    if current_awake_streak >= 1800 and not achievements["no_drowsy_30min"]:
        achievements["no_drowsy_30min"] = True
        play_voice_alert("Achievement unlocked: Thirty minutes alert!")
    
    # Blink master (high blink rate)
    if total_blinks > 100 and not achievements["blink_master"]:
        achievements["blink_master"] = True
        play_voice_alert("Achievement unlocked: Blink master!")
    
    # Marathon driver (2 hours)
    if duration >= 7200 and not achievements["marathon_driver"]:
        achievements["marathon_driver"] = True
        play_voice_alert("Achievement unlocked: Marathon driver!")

def draw_ear_graph(frame):
    """Draw real-time EAR graph"""
    if len(ear_history) < 2:
        return
    
    graph_x, graph_y = frame.shape[1] - 250, 50
    graph_w, graph_h = 200, 100
    
    # Draw graph background
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
    
    # Draw threshold line
    threshold_y = graph_y + graph_h - int((EAR_THRESHOLD / 0.5) * graph_h)
    cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), (0, 0, 255), 1)
    
    # Draw EAR values
    points = []
    for i, ear in enumerate(ear_history):
        x = graph_x + int((i / len(ear_history)) * graph_w)
        y = graph_y + graph_h - int((ear / 0.5) * graph_h)
        y = max(graph_y, min(graph_y + graph_h, y))
        points.append((x, y))
    
    for i in range(1, len(points)):
        cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
    
    cv2.putText(frame, "EAR Graph", (graph_x, graph_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_enhanced_status(frame, avg_ear, face_rect):
    global awake_status, alert_level, current_awake_streak, longest_awake_streak
    
    # Update awake streak
    if alert_level == 0:
        current_awake_streak += 1
        if current_awake_streak > longest_awake_streak:
            longest_awake_streak = current_awake_streak
    else:
        current_awake_streak = 0
    
    # Calculate statistics
    duration = get_session_duration()
    minutes = duration // 60
    seconds = duration % 60
    blinks_per_minute = (total_blinks / max(duration, 1)) * 60
    drowsy_percentage = (total_drowsy_time / max(duration, 1)) * 100
    
    # Color scheme for night mode
    text_color = (100, 100, 100) if night_mode else (255, 255, 255)
    bg_color = (30, 30, 30) if night_mode else (0, 0, 0)
    
    # Draw semi-transparent background for better readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 300), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Session info
    cv2.putText(frame, f"Session: {minutes}m {seconds}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"EAR: {avg_ear:.3f} (Threshold: {EAR_THRESHOLD:.3f})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f"FPS: {current_fps}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Statistics
    cv2.putText(frame, f"Drowsy Episodes: {drowsy_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Blinks/min: {blinks_per_minute:.1f}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    cv2.putText(frame, f"Drowsy Time: {total_drowsy_time:.1f}s ({drowsy_percentage:.1f}%)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Streaks
    cv2.putText(frame, f"Current Streak: {current_awake_streak//30}s", (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f"Best Streak: {longest_awake_streak//30}s", (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Alert status with animation
    if alert_level == 0:
        status_color = (0, 255, 0)
        awake_status = "AWAKE"
    elif alert_level == 1:
        status_color = (0, 255, 255)
        awake_status = "WARNING"
    else:
        # Animated red alert
        intensity = int(128 + 127 * abs(np.sin(time.time() * 5)))
        status_color = (0, 0, intensity)
        awake_status = "DANGER"
    
    cv2.putText(frame, awake_status, (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    # Enhanced drowsiness meter
    draw_advanced_meter(frame, avg_ear)
    
    # Face detection confidence
    if face_rect is not None:
        cv2.putText(frame, f"Detection: {detection_confidence:.0f}%", (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Achievement indicators
    draw_achievements(frame)

def draw_advanced_meter(frame, avg_ear):
    """Draw advanced drowsiness meter with gradient"""
    bar_width = 300
    bar_height = 25
    bar_x, bar_y = 10, 300
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Calculate fill percentage
    if frame_counter > 0:
        fill_percentage = min(frame_counter / EAR_CONSEC_FRAMES, 1.0)
        fill_width = int(fill_percentage * bar_width)
        
        # Gradient colors
        for i in range(fill_width):
            color_ratio = i / bar_width
            if color_ratio < 0.5:
                color = (0, int(255 * (1 - color_ratio * 2)), int(255 * color_ratio * 2))
            else:
                color = (0, 0, int(255 * (2 - color_ratio * 2)))
            cv2.line(frame, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height), color, 1)
    
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.putText(frame, "Drowsiness Level", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_achievements(frame):
    """Draw achievement indicators"""
    y_offset = 50
    for achievement, unlocked in achievements.items():
        if unlocked:
            cv2.putText(frame, f"🏆 {achievement.replace('_', ' ').title()}", (frame.shape[1] - 300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25

def draw_face_detection_box(frame, face_rect):
    """Draw animated face detection box"""
    if face_rect is not None:
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        
        # Animated corners
        corner_length = 20
        thickness = 3
        color = (0, 255, 0) if alert_level == 0 else (0, 0, 255)
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)

def save_session_data():
    """Save detailed session data"""
    data = {
        "session_duration": get_session_duration(),
        "drowsy_count": drowsy_count,
        "total_blinks": total_blinks,
        "total_drowsy_time": total_drowsy_time,
        "longest_awake_streak": longest_awake_streak,
        "achievements": achievements,
        "ear_threshold": EAR_THRESHOLD,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open("logs/session_data.json", "w") as f:
        json.dump(data, f, indent=2)

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

print("🚗 ADVANCED DROWSINESS DETECTION SYSTEM 🚗")
print("=" * 50)
print("Controls:")
print("'q' - Quit")
print("'s' - Save screenshot")
print("'r' - Reset counters")
print("'m' - Toggle sound on/off")
print("'n' - Toggle night mode")
print("'l' - Toggle landmarks display")
print("'p' - Pause/Resume detection")
print("'+/-' - Adjust sensitivity")
print("=" * 50)

# Variables for blink detection
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
        cv2.imshow("Advanced Drowsiness Detector", frame)
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

    for face in faces:
        face_detected = True
        current_face_rect = face
        detection_confidence = 85.0  # Simulated confidence
        
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [shape[i] for i in LEFT_EYE]
        right_eye = [shape[i] for i in RIGHT_EYE]

        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Add to EAR history
        ear_history.append(avg_ear)

        # Blink detection
        if previous_ear > EAR_THRESHOLD and avg_ear < EAR_THRESHOLD:
            if not blink_detected:
                total_blinks += 1
                blink_detected = True
                last_blink_time = time.time()
        elif avg_ear > EAR_THRESHOLD:
            blink_detected = False
        
        previous_ear = avg_ear

        # Enhanced drowsiness logic
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            
            # Warning level
            if frame_counter >= EAR_CONSEC_FRAMES // 2:
                alert_level = 1
                if frame_counter == EAR_CONSEC_FRAMES // 2:
                    play_warning_sound()
            
            # Danger level
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

        # Draw face detection box
        draw_face_detection_box(frame, current_face_rect)

        # Draw eye landmarks (if enabled)
        if show_landmarks:
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Break reminder every 30 minutes
    if time.time() - last_break_reminder > 1800:  # 30 minutes
        play_voice_alert("Break reminder: Consider taking a rest")
        last_break_reminder = time.time()

    # Check achievements
    check_achievements()

    # Draw all enhanced features
    draw_enhanced_status(frame, avg_ear if face_detected else 0, current_face_rect)
    draw_ear_graph(frame)

    # Display help text
    help_text = "q:quit | s:screenshot | r:reset | m:sound | n:night | l:landmarks | p:pause | +/-:sensitivity"
    cv2.putText(frame, help_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Advanced Drowsiness Detector", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/manual_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    elif key == ord('r'):
        drowsy_count = 0
        total_blinks = 0
        session_start_time = time.time()
        total_drowsy_time = 0
        longest_awake_streak = 0
        current_awake_streak = 0
        ear_history.clear()
        print("🔄 Counters reset!")
    elif key == ord('m'):
        sound_enabled = not sound_enabled
        print(f"🔊 Sound: {'ON' if sound_enabled else 'OFF'}")
        if not sound_enabled:
            stop_alarm()
    elif key == ord('n'):
        night_mode = not night_mode
        print(f"🌙 Night mode: {'ON' if night_mode else 'OFF'}")
    elif key == ord('l'):
        show_landmarks = not show_landmarks
        print(f"👁️ Landmarks: {'ON' if show_landmarks else 'OFF'}")
    elif key == ord('p'):
        system_paused = not system_paused
        print(f"⏸️ System: {'PAUSED' if system_paused else 'RESUMED'}")
    elif key == ord('+') or key == ord('='):
        EAR_THRESHOLD = min(0.35, EAR_THRESHOLD + 0.01)
        print(f"🎯 Sensitivity increased: {EAR_THRESHOLD:.3f}")
    elif key == ord('-'):
        EAR_THRESHOLD = max(0.15, EAR_THRESHOLD - 0.01)
        print(f"🎯 Sensitivity decreased: {EAR_THRESHOLD:.3f}")

# Save session data
save_session_data()

# Print enhanced session summary
print("\n" + "="*50)
print("📊 ENHANCED SESSION SUMMARY")
print("="*50)
print(f"⏱️  Total Duration: {get_session_duration()//60}m {get_session_duration()%60}s")
print(f"😴 Drowsy Episodes: {drowsy_count}")
print(f"👁️  Total Blinks: {total_blinks}")
print(f"💤 Total Drowsy Time: {total_drowsy_time:.1f}s")
print(f"🏆 Longest Awake Streak: {longest_awake_streak//30}s")
print(f"📈 Average FPS: {current_fps}")
print(f"🎯 Final Threshold: {EAR_THRESHOLD:.3f}")
print(f"🏅 Achievements Unlocked: {sum(achievements.values())}/4")
print(f"💾 Data saved to 'logs' directory")
print("="*50)

# Cleanup
stop_alarm()
cap.release()
cv2.destroyAllWindows()
voice_engine.stop()