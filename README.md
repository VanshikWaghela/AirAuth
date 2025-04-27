AirAuth: Gesture Control System

# 🔒 **Touchless security meets intuitive control.**
Control your computer using real-time hand gestures, but only after securely authenticating your face.

AirAuth is a sophisticated macOS-compatible webcam-based gesture control system that allows you to control your computer using hand gestures. Designed with a focus on intuitive interactions, smooth transitions, and visual feedback, CV_Mouse aims to provide a natural and delightful user experience. The system now includes a face recognition security gate that only enables gestures when a recognized user is present.

## 🚀 Setup Instructions (macOS)

1. **Prerequisites:**
   - macOS operating system
   - Webcam access
   - Conda environment
   - Accessibility and Camera permissions granted

2. **Installation:**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/CV_Mouse.git
   cd CV_Mouse

   # Activate the conda environment
   conda activate vis

   # Install dependencies (if needed)
   pip install -r requirements.txt

   # Create faces directory if it doesn't exist
   mkdir -p faces
   ```

3. **Running the Application:**
   ```bash
   # Enroll a user (required before first use)
   python app.py --enroll username

   # Start the gesture control system
   python app.py

   # Enable debug mode (shows face distance metrics)
   python app.py --debug
   ```

## 🔒 Face Recognition Gate

The system now includes a face recognition security layer that only enables gesture control when a recognized user is present:

1. **User Enrollment:**
   - Run `python app.py --enroll username` to register a new user
   - Follow the on-screen prompts to capture 5 face images
   - Press spacebar to capture each image when your face is detected
   - Press ESC to exit enrollment mode

2. **Authentication:**
   - When the application starts, it will remain locked until a recognized face is detected
   - A red banner with "🔒 Access Denied" indicates the locked state
   - When a recognized user is detected, a green banner with the username appears
   - Gestures are only active in the unlocked state

3. **Configuration:**
   - Edit `config.yaml` to adjust face recognition settings:
     - `model`: "hog" (faster) or "cnn" (more accurate, requires GPU)
     - `tolerance`: Lower values (e.g., 0.45) are more strict, higher values (e.g., 0.65) are more permissive
     - `consecutive_frames`: Number of frames required to change lock/unlock state
     - `enroll_snapshots`: Number of images to capture during enrollment

## 🖐 Gesture Commands Reference

| Gesture | Description | Function |
|---------|-------------|----------|
| **Index finger extended** | Extend only your index finger | Move the cursor |
| **Thumb-Index pinch** | Touch your thumb and index finger together | Left mouse click |
| **Thumb-Middle finger pinch** | Touch your thumb and middle finger together | Right mouse click |
| **Victory sign (V)** | Hold steady V-sign with index and middle fingers | Toggle drawing mode on/off |
| **Index finger in drawing mode** | Move index finger when drawing mode is active | Draw on screen |
| **Open palm up/down** | All fingers extended, move hand up/down | Scroll up/down |
| **Two-finger swipe left** | Index and middle finger extended, swipe left | Switch to previous tab |
| **Two-finger swipe right** | Index and middle finger extended, swipe right | Switch to next tab |
| **Index finger + wrist rotation** | Only index extended, rotate wrist | Adjust system volume |
| **Two-hand pinch in/out** | Both hands pinching, move apart/together | Zoom in/out |

## ⌨️ Keyboard Controls

| Key | Function |
|-----|----------|
| `s` | Save drawing (when in drawing mode) |
| `q` or `Esc` | Quit application |

## 🎯 Calibration Guide

For optimal performance, follow these calibration steps when starting the application:

1. **Positioning:**
   - Sit approximately 50-70cm from your webcam
   - Ensure your hands are clearly visible in the frame
   - Use a neutral background for better hand detection

2. **Lighting:**
   - Use consistent, even lighting
   - Avoid strong backlighting or shadows on your hands

3. **Initial Hand Pose:**
   - Begin with an open palm facing the camera
   - Move your hand slowly at first to allow the system to track it properly

4. **Gesture Practice:**
   - Practice each gesture while observing the on-screen feedback
   - Pay attention to the UI overlay which shows the current detected gesture

## 🖐️ Gesture Controls

1. **Basic Controls:**
   - **Cursor Movement**: Index finger extended
   - **Left Click**: Pinch with thumb and index finger (other fingers closed)
   - **Right Click**: Pinch with thumb and middle finger (ring and pinky closed)
   - **Scroll**: Open palm with upward/downward movement

2. **Mode Controls:**
   - **Drawing Mode Toggle**: V-sign (index and middle fingers extended)
   - **Play/Pause Media**: Closed fist gesture

3. **Two-handed Gestures:**
   - **Zoom In/Out**: Move both hands apart/together
   - **Tab Switching**: Two-finger horizontal swipe (both hands moving in same direction)

## 💡 Tips for Best Performance

1. **Hand Visibility:**
   - Keep your hands within the camera frame at all times
   - Avoid rapid movements that might cause tracking loss
   - Maintain a consistent distance from the camera

2. **Gesture Execution:**
   - Make deliberate, clear gestures
   - Hold gestures steady for a moment for better recognition
   - For cursor control, use small movements for precision

3. **System Resources:**
   - Close resource-intensive applications for smoother performance
   - Monitor the FPS indicator in the UI overlay
   - If performance drops, try reducing other background processes

4. **Drawing Mode:**
   - Use slow, deliberate movements for precise drawing
   - Press 's' to save your drawing to the Desktop
   - Use the V-sign gesture to toggle drawing mode on/off

5. **Two-handed Gestures:**
   - Ensure both hands are clearly visible in the frame
   - Keep hand movements synchronized for zoom gestures
   - For tab switching, move both hands in the same horizontal direction

*This documentation is automatically updated when changes are made to the gesture control system.*

Last updated: May 15, 2023
