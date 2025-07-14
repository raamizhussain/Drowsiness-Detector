# Face Landmarks Detection System

## Overview
A real-time face landmarks detection system using OpenCV and dlib that identifies and visualizes 68 facial key points on a live video feed.

## FIND DEMONSTRATION VIDEO IN THE REPOSITORY (https://github.com/raamizhussain/Drowsiness-Detector/), COULDN'T FIT IT HERE IN THE README file

## Key Features

### 🎯 **Face Detection**
- Uses dlib's robust frontal face detector
- Processes grayscale frames for optimal performance
- Handles multiple faces simultaneously

### 📍 **68-Point Landmark Detection**
- **Eyes**: Points 36-47 (left: 36-41, right: 42-47)
- **Eyebrows**: Points 17-26 (left: 17-21, right: 22-26)
- **Nose**: Points 27-35 (bridge: 27-30, nostrils: 31-35)
- **Mouth**: Points 48-67 (outer: 48-59, inner: 60-67)
- **Jaw**: Points 0-16 (chin to temples)

### 🎨 **Visual Feedback**
- Green circles mark each landmark point
- Real-time overlay on video feed
- 2-pixel radius circles for clear visibility

### ⚡ **Performance**
- Efficient grayscale processing
- Real-time video capture at webcam FPS
- Minimal computational overhead

## Controls
- **'q'**: Quit application
- **ESC**: Alternative exit method

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

### 🔧 **Enhanced Features**
- **Selective Landmark Display**: Toggle eye/mouth/nose points individually
- **Landmark Numbering**: Show point indices for development
- **Distance Measurements**: Calculate eye-to-eye, mouth width ratios
- **Confidence Scoring**: Display detection confidence levels

### 📊 **Advanced Analytics**
- **Facial Expression Recognition**: Classify emotions based on landmark positions
- **Head Pose Estimation**: Calculate pitch, yaw, roll angles
- **Eye Gaze Tracking**: Determine looking direction
- **Blink Detection**: Monitor eye closure patterns

### 🎨 **Visual Enhancements**
- **Customizable Colors**: User-selectable landmark colors
- **Connection Lines**: Draw facial feature outlines
- **Region Highlighting**: Color-code different facial areas
- **3D Visualization**: Project landmarks in 3D space

### ⚡ **Performance Optimizations**
- **Multi-threading**: Separate detection and display threads
- **GPU Acceleration**: CUDA support for faster processing
- **Adaptive Resolution**: Dynamic quality adjustment
- **Memory Management**: Efficient frame buffering

### 🔒 **Robustness Features**
- **Profile Face Support**: Side-view detection
- **Low-light Enhancement**: Automatic brightness adjustment
- **Motion Blur Handling**: Stabilization algorithms
- **Multi-face Tracking**: Persistent face IDs

### 💾 **Data & Export**
- **Recording Capability**: Save annotated video
- **CSV Export**: Landmark coordinates over time
- **Calibration System**: Personal face model training
- **JSON Output**: Structured landmark data export

### 🎯 **Application Integration**
- **Drowsiness Detection**: Eye aspect ratio monitoring
- **Attention Tracking**: Focus measurement systems
- **Accessibility Tools**: Facial control interfaces
- **Security Systems**: Face recognition integration

## Dependencies
- **OpenCV**: Computer vision operations
- **dlib**: Face detection and landmark prediction
- **NumPy**: Numerical computations (implicit)

## Hardware Requirements
- **Webcam**: Any USB/built-in camera
- **CPU**: Modern processor for real-time processing
- **RAM**: Minimum 4GB recommended
- **Storage**: ~100MB for model files
