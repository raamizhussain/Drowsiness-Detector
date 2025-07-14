# Drowsiness Detection System

## Overview
This Drowsiness Detection System is a real-time monitoring application that uses your webcam to track 68 facial key points and analyze your alertness levels. It continuously measures your Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect drowsiness, blinks, and yawns, providing instant visual and audio alerts when it detects signs of fatigue. The system displays live graphs showing your eye and mouth activity, maintains session statistics, and can save screenshots and logs of detection events - making it ideal for driver monitoring, student attention tracking, or any scenario where maintaining alertness is critical.

DEMO VIDEO LINK: https://drive.google.com/file/d/1LZ5RkTWKtUtyFxq2_HCEXTGEuzSnixiZ/view?usp=sharing

## Key Features

### 🎯 **Face Detection**
- Uses dlib's robust frontal face detector
- Processes grayscale frames for optimal performance
- Handles multiple faces simultaneously
- Corner-based detection box with dynamic colors

### 📍 **68-Point Landmark Detection**
- **Eyes**: Points 36-47 (left: 36-41, right: 42-47) - Green circles
- **Eyebrows**: Points 17-26 (left: 17-21, right: 22-26)
- **Nose**: Points 27-35 (bridge: 27-30, nostrils: 31-35)
- **Mouth**: Points 48-67 (outer: 48-59, inner: 60-67) - Yellow circles
- **Jaw**: Points 0-16 (chin to temples)

### 📊 **Real-Time Analytics**
- **EAR (Eye Aspect Ratio)**: Measures eye openness (0.0-0.5 range)
- **MAR (Mouth Aspect Ratio)**: Detects yawns (0.0-1.0 range)
- **Blink Detection**: Tracks eye closure patterns
- **Drowsiness Monitoring**: Consecutive frame analysis
- **Yawn Detection**: Mouth opening threshold monitoring

### 📈 **Interactive Graphs**
- **EAR Graph**: Real-time eye aspect ratio plotting
  - Green line shows current EAR values
  - Red threshold line at 0.25 for drowsiness detection
  - 100-point history buffer for smooth visualization
- **MAR Graph**: Live mouth aspect ratio tracking
  - Yellow line displays mouth opening levels
  - Red threshold line at 0.5 for yawn detection
  - Scaled 0.0-1.0 range for optimal viewing

### 🎨 **Enhanced Visual Interface**
- **Status Panel**: Session duration, statistics, FPS counter
- **Alert System**: Color-coded status (Green: Awake, Yellow: Warning, Red: Danger)
- **Achievement System**: Unlockable milestones with voice notifications
- **Night Mode**: Dark theme for low-light environments
- **Landmark Toggle**: Show/hide facial feature points

### ⚡ **Performance Monitoring**
- Real-time FPS calculation and display
- Detection confidence scoring
- Memory-efficient circular buffers
- Optimized grayscale processing

## Controls & Features
- **'q'**: Quit application
- **'s'**: Take screenshot and save to logs/
- **'r'**: Reset all counters and statistics
- **'m'**: Toggle sound alerts on/off
- **'n'**: Toggle night mode (dark theme)
- **'l'**: Toggle landmark visibility
- **'p'**: Pause/Resume system
- **'+/-'**: Adjust EAR/MAR sensitivity thresholds

## Alert System
- **Voice Alerts**: Text-to-speech warnings for drowsiness/yawns
- **Sound Effects**: Configurable WAV files (alarm.wav, warning.wav, yawn.wav)
- **Visual Warnings**: Screen flash effects and color changes
- **Progressive Alerts**: Escalating warnings based on severity

## Technical Implementation

### Core Components
1. **Face Detection**: `dlib.get_frontal_face_detector()`
2. **Landmark Prediction**: `shape_predictor_68_face_landmarks.dat`
3. **Video Processing**: OpenCV VideoCapture
4. **Visualization**: Circle overlays on detected points

### Processing Pipeline
```
Video Frame → Grayscale → Face Detection → Landmark Prediction → Visualization → Display
```

## Future Improvements

### 🔧 **Core Enhancements**
- **Multi-angle Detection**: Profile and side-view face support
- **Emotion Recognition**: Classify facial expressions from landmarks
- **Head Pose Estimation**: Calculate 3D head orientation
- **Eye Gaze Tracking**: Determine looking direction

### 📊 **Advanced Analytics**
- **Fatigue Scoring**: Comprehensive drowsiness assessment
- **Attention Metrics**: Focus and engagement measurement
- **Biometric Integration**: Heart rate from facial color changes
- **Personalized Baselines**: Individual user calibration

### 🎯 **Practical Applications**
- **Driver Monitoring**: Automotive drowsiness detection
- **Student Attention**: Educational engagement tracking
- **Medical Monitoring**: Patient alertness assessment
- **Security Systems**: Access control integration

## Dependencies
- **OpenCV**: Computer vision operations and video processing
- **dlib**: Face detection and 68-point landmark prediction
- **pygame**: Sound system and audio alerts
- **pyttsx3**: Text-to-speech voice alerts
- **scipy**: Mathematical calculations for EAR/MAR
- **numpy**: Numerical computations and array operations

## Data Logging & Storage
- **logs/drowsiness_log.txt**: Timestamped drowsiness episodes
- **logs/yawn_log.txt**: Yawn detection records
- **logs/session_data.json**: Complete session statistics
- **logs/screenshots/**: Manual and automatic image captures

## Hardware Requirements
- **Webcam**: Any USB/built-in camera
- **CPU**: Modern processor for real-time processing
- **RAM**: Minimum 4GB recommended
- **Storage**: ~100MB for model files
