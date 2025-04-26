#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py: Modular, robust gesture detection and smoothing for CV_Mouse.
All gesture detection is handled here, with FSM/debounce for reliability.
"""
import numpy as np
import cv2
import mediapipe as mp
import time
import subprocess
import pyautogui
import yaml

def load_config(path):
    """Load configuration from YAML file with defaults"""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}. Using defaults.")
        # Return minimal default configuration
        return {
            'smoothing': {'method': 'one_euro', 'params': {'mincutoff': 1.5, 'beta': 0.01}},
            'detection': {'min_confidence': 0.8},
            'gestures': {
                'left_click': {'hold_frames': 6, 'debounce_s': 0.4},
                'right_click': {'hold_frames': 6, 'debounce_s': 0.4},
                'scroll': {'dy_thresh': 60, 'dt_max': 0.5},
                'tab_switch': {'hold_frames': 6, 'dt_max': 0.5},
                'cooldown_s': 0.4
            },
            'mac_keys': {
                'tab_forward': ["command", "tab"],
                'tab_back': ["command", "shift", "tab"],
                'zoom_in': ["command", "+"],
                'zoom_out': ["command", "-"]
            },
            'ui': {'show_hand_landmarks': True, 'show_fps': True},
            'face_recognition': {
                'model': "hog",
                'tolerance': 0.55,
                'consecutive_frames': 3,
                'enroll_snapshots': 5,
                'folder_path': "faces/"
            }
        }

mp_hands = mp.solutions.hands

# --- Smoothing Filters ---
class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2, error_cov=1e-1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.error_cov = error_cov
        self.x = None
        self.p = None
    def apply(self, measurement):
        if self.x is None:
            self.x = np.array(measurement, dtype=np.float32)
            self.p = np.ones_like(self.x) * self.error_cov
            return self.x.copy()
        x_pred = self.x.copy()
        p_pred = self.p + self.process_noise
        k = p_pred / (p_pred + self.measurement_noise)
        self.x = x_pred + k * (np.array(measurement) - x_pred)
        self.p = (1 - k) * p_pred
        return self.x.copy()

class OneEuroFilter:
    """
    Enhanced 1€ filter for smooth, responsive signal filtering.
    Reference: Casiez, G., Roussel, N., & Vogel, D. (2012).
    Optimized version with improved responsiveness and stability.
    """
    def __init__(self, freq=120.0, mincutoff=0.5, beta=0.15, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.last_time = None
        # Pre-calculate constants where possible
        self._te = 1.0 / self.freq
        self._a_d = self._calculate_alpha(self.dcutoff)
        # Add smoothing history for improved stability
        self.history = []
        self.history_size = 3  # Small history size for responsiveness
        
    def _calculate_alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self._te)
        
    def filter(self, x, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        if self.x_prev is None:
            self.last_time = now
            self.x_prev = x
            return x
            
        # Update frequency based on actual time elapsed with improved smoothing
        dt = now - self.last_time
        if dt > 0:
            # More responsive frequency adaptation
            self.freq = 0.8 * self.freq + 0.2 * (1.0 / dt)  # More weight to current frame rate
            self._te = 1.0 / self.freq
            self._a_d = self._calculate_alpha(self.dcutoff)
        
        # Calculate derivative with improved noise handling
        dx = (x - self.x_prev) * self.freq
        
        # Filter derivative with improved parameters
        dx_hat = self._a_d * dx + (1 - self._a_d) * self.dx_prev
        
        # Enhanced adaptive cutoff frequency based on movement speed
        # More responsive for small movements, more stable for large ones
        movement_magnitude = abs(dx_hat)
        cutoff = self.mincutoff
        
        # Progressive cutoff adjustment based on movement speed
        if movement_magnitude < 1.0:
            # Very small movements - be more responsive
            cutoff = self.mincutoff
        elif movement_magnitude < 5.0:
            # Medium movements - moderate filtering
            cutoff = self.mincutoff + self.beta * movement_magnitude * 0.5
        else:
            # Fast movements - stronger filtering to reduce jitter
            cutoff = self.mincutoff + self.beta * movement_magnitude
        
        # Calculate alpha for current cutoff
        a = self._calculate_alpha(cutoff)
        
        # Filter position with improved algorithm
        x_hat = a * x + (1 - a) * self.x_prev
        
        # Add to history for optional additional smoothing
        self.history.append(x_hat)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        
        # Apply minimal additional smoothing if we have enough history
        # This helps reduce micro-jitters while maintaining responsiveness
        if len(self.history) == self.history_size:
            # Weighted average with more weight to recent values
            weights = [0.1, 0.3, 0.6]  # More weight to recent values
            x_hat = sum(w * v for w, v in zip(weights, self.history)) / sum(weights)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.last_time = now
        
        return x_hat
        
    def apply(self, x):
        return self.filter(x)

# --- Gesture FSM/State ---
class GestureFSM:
    def __init__(self, hold_frames=3):
        self.frames = 0
        self.active = False
        self.hold_frames = hold_frames
    def update(self, detected):
        if detected:
            self.frames += 1
            if self.frames >= self.hold_frames:
                self.active = True
        else:
            self.frames = 0
            self.active = False
        return self.active

# --- Utility Functions ---
def euclidean(p1, p2):
    """Optimized Euclidean distance calculation"""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return np.sqrt(dx*dx + dy*dy + dz*dz)
    
# Fast version that avoids square root for comparisons
def squared_distance(p1, p2):
    """Faster distance calculation that avoids square root"""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return dx*dx + dy*dy + dz*dz

def get_finger_states(hand):
    lm = hand.landmark
    # Thumb: tip.x < ip.x (right hand), > for left hand
    thumb = int(lm[4].x < lm[3].x) if lm[17].x > lm[5].x else int(lm[4].x > lm[3].x)
    index = int(lm[8].y < lm[6].y)
    middle = int(lm[12].y < lm[10].y)
    ring = int(lm[16].y < lm[14].y)
    pinky = int(lm[20].y < lm[18].y)
    return [thumb, index, middle, ring, pinky]

def detect_pinch(hand_landmarks, which='index'):
    """
    Optimized pinch gesture detection between thumb and specified finger
    Args:
        hand_landmarks: MediaPipe hand landmarks
        which: 'index' or 'middle' finger to detect pinch with
    Returns:
        bool: True if pinch detected
    """
    if hand_landmarks is None:
        return False

    # Get landmark indices based on finger type (pre-computed for speed)
    thumb_idx = 4  # Thumb tip is always at index 4
    finger_idx = 8 if which == 'index' else 12 if which == 'middle' else -1
    
    if finger_idx == -1:
        return False
        
    # Direct access to landmarks for better performance
    thumb_tip = hand_landmarks.landmark[thumb_idx]
    finger_tip = hand_landmarks.landmark[finger_idx]

    # Faster distance calculation (avoid sqrt for speed)
    # We can compare squared distance with squared threshold
    dx = thumb_tip.x - finger_tip.x
    dy = thumb_tip.y - finger_tip.y
    dz = thumb_tip.z - finger_tip.z
    squared_dist = dx*dx + dy*dy + dz*dz
    
    # Threshold for pinch detection (0.05 squared = 0.0025)
    return squared_dist < 0.0025

def detect_index_only(hand):
    fs = get_finger_states(hand)
    return fs == [0,1,0,0,0]

def detect_scroll(hand, prev_hand, thresh=0.01):
    fs = get_finger_states(hand)
    # Accept both [0,1,1,0,0] and [1,1,1,0,0] finger states for more reliable detection
    if fs == [0,1,1,0,0] or fs == [1,1,1,0,0]:
        # Use palm position (landmark 0) for more stable scrolling
        dy = hand.landmark[0].y - prev_hand.landmark[0].y
        # Use wrist movement for scrolling direction
        # Apply a small amount of smoothing to prevent jitter
        if abs(dy) > thresh:
            # Scale the scrolling speed based on movement magnitude for better control
            scroll_speed = min(5, max(1, int(abs(dy) * 100)))
            return True, -np.sign(dy) * scroll_speed
    return False, 0

def detect_tab_switch(hand, prev_hand, thresh=0.08):
    fs = get_finger_states(hand)
    # Check for V sign (index and middle fingers extended)
    if fs[1] == 1 and fs[2] == 1 and sum(fs[3:]) == 0:  # More flexible detection
        # Use palm position for more stable detection
        dx = hand.landmark[0].x - prev_hand.landmark[0].x
        # Check if horizontal movement exceeds threshold
        if abs(dx) > thresh:
            # Return direction (true for right, false for left)
            return True, dx > 0
    return False, False

def detect_volume(hand, prev_hand, thresh=0.15):
    def angle(lm):
        v = np.array([lm[17].x-lm[5].x, lm[17].y-lm[5].y])
        return np.arctan2(v[1], v[0])
    a_now = angle(hand.landmark)
    a_prev = angle(prev_hand.landmark)
    delta = a_now - a_prev
    if abs(delta) > thresh:
        return True, delta < 0
    return False, False

def detect_zoom(hand1, hand2, prev1, prev2, thresh=0.08):
    """Optimized zoom gesture detection using squared distances"""
    # Use squared distances for faster comparison
    d_now_sq = squared_distance(hand1.landmark[8], hand2.landmark[8])
    d_prev_sq = squared_distance(prev1.landmark[8], prev2.landmark[8])
    
    # Take square root only when needed for the comparison
    d_now = np.sqrt(d_now_sq)
    d_prev = np.sqrt(d_prev_sq)
    
    # Fast comparison
    if abs(d_now-d_prev) > thresh:
        return True, d_now > d_prev
    return False, False

def detect_fist(hand):
    return get_finger_states(hand) == [0,0,0,0,0]

def detect_neutral(hand):
    return get_finger_states(hand) == [1,1,1,1,1]

def detect_victory(hand, min_dist=0.08, min_angle=18):
    fs = get_finger_states(hand)
    if not (fs[1] == 1 and fs[2] == 1 and fs[0] == 0):
        return False
    lm = hand.landmark
    tip_dist = np.sqrt((lm[8].x-lm[12].x)**2 + (lm[8].y-lm[12].y)**2)
    if tip_dist < min_dist:
        return False
    v1 = np.array([lm[8].x-lm[5].x, lm[8].y-lm[5].y])
    v2 = np.array([lm[12].x-lm[9].x, lm[12].y-lm[9].y])
    dot = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    if angle < min_angle:
        return False
    return True

def toggle_media_playback():
    try:
        subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 100'], check=True)
    except Exception as e:
        print(f"[WARN] Media toggle failed: {e}")

def adjust_volume(direction, step=2):
    try:
        get_vol = subprocess.run(['osascript', '-e', 'output volume of (get volume settings)'], capture_output=True, text=True)
        vol = int(get_vol.stdout.strip())
        new_vol = max(0, min(100, vol + (step if direction=='up' else -step)))
        subprocess.run(['osascript', '-e', f'set volume output volume {new_vol}'])
        return new_vol
    except Exception as e:
        print(f"[WARN] Volume adjust failed: {e}")
        return None

# --- Face Recognition Functions ---
import os
import threading
from typing import Dict, List, Tuple, Optional
import face_recognition

# Cache for face encodings to avoid redundant calculations
face_encoding_cache = {}

def load_face_data(folder_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Scan each subfolder under folder_path,
    load each image, detect face, compute embedding,
    return { username: [emb1, emb2, …] }.
    """
    face_data = {}
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return face_data
    
    # Scan each user subfolder
    for username in os.listdir(folder_path):
        user_folder = os.path.join(folder_path, username)
        if not os.path.isdir(user_folder) or username.startswith('.'):
            continue
            
        embeddings = []
        for filename in os.listdir(user_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(user_folder, filename)
                try:
                    # Check if encoding is already cached
                    if image_path in face_encoding_cache:
                        embeddings.append(face_encoding_cache[image_path])
                        continue
                        
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        # Cache the encoding
                        face_encoding_cache[image_path] = face_encodings[0]
                        embeddings.append(face_encodings[0])
                except Exception as e:
                    print(f"[ERROR] Could not process {image_path}: {e}")
        
        if embeddings:
            face_data[username] = embeddings
    
    return face_data

# Thread-safe variables for face recognition
class FaceRecognitionState:
    def __init__(self):
        self.result = (None, 1.0)
        self.frame_to_process = None
        self.known_embeddings = {}
        self.cfg = None
        self.lock = threading.Lock()
        self.processing = False
        self.frame_count = 0
        self.skip_frames = 3  # Process every 4th frame

face_state = FaceRecognitionState()

def face_recognition_thread():
    """Background thread for face recognition processing"""
    while True:
        # Check if there's a frame to process
        with face_state.lock:
            if face_state.frame_to_process is None or not face_state.known_embeddings or not face_state.cfg:
                face_state.processing = False
                time.sleep(0.01)  # Short sleep to prevent CPU spinning
                continue
                
            frame = face_state.frame_to_process.copy()
            known_embeddings = face_state.known_embeddings
            cfg = face_state.cfg
            face_state.frame_to_process = None
            face_state.processing = True
        
        # Process the frame (outside the lock)
        result = recognize_face_internal(frame, known_embeddings, cfg)
        
        # Update the result
        with face_state.lock:
            face_state.result = result
            face_state.processing = False

# Start the face recognition thread
face_thread = threading.Thread(target=face_recognition_thread, daemon=True)
face_thread.start()

def recognize_face(frame: np.ndarray, known_embeddings: Dict[str, List[np.ndarray]], cfg) -> Tuple[Optional[str], float]:
    """Thread-safe wrapper for face recognition"""
    with face_state.lock:
        # Skip frames to improve performance
        face_state.frame_count = (face_state.frame_count + 1) % (face_state.skip_frames + 1)
        if face_state.frame_count != 0:
            return face_state.result
            
        # If not currently processing, submit a new frame
        if not face_state.processing and face_state.frame_to_process is None:
            face_state.frame_to_process = frame
            face_state.known_embeddings = known_embeddings
            face_state.cfg = cfg
            
        return face_state.result

def recognize_face_internal(frame: np.ndarray, known_embeddings: Dict[str, List[np.ndarray]], cfg) -> Tuple[Optional[str], float]:
    """
    Internal function for face recognition processing
    """
    # No known faces to compare against
    if not known_embeddings:
        return None, 1.0
    
    # Resize frame for faster processing (more aggressive downsampling)
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)  # 20% of original size
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find faces in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Always use HOG for speed
    if not face_locations:
        return None, 1.0
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    if not face_encodings:
        return None, 1.0
    
    # Compare with known faces
    best_match = None
    min_distance = float('inf')
    
    for username, stored_encodings in known_embeddings.items():
        for stored_encoding in stored_encodings:
            # Calculate face distance
            face_distance = face_recognition.face_distance([stored_encoding], face_encodings[0])[0]
            
            if face_distance < min_distance:
                min_distance = face_distance
                best_match = username
    
    # Return match if distance is below tolerance
    tolerance = cfg['face_recognition']['tolerance']
    if min_distance <= tolerance:
        return best_match, min_distance
    else:
        return None, min_distance

def enroll_user(username: str, cap: cv2.VideoCapture, cfg) -> None:
    """
    - Prompt: "Look at camera…"
    - Capture cfg.face_recognition.enroll_snapshots frames.
    - Save aligned face crops to faces/username/.
    - Compute & save embeddings as .npy alongside images.
    """
    folder_path = cfg['face_recognition']['folder_path']
    user_folder = os.path.join(folder_path, username)
    
    # Create folder if it doesn't exist
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    snapshots_count = cfg['face_recognition']['enroll_snapshots']
    captured = 0
    
    print(f"[INFO] Enrolling user {username}. Please look at the camera...")
    
    # For automatic capture timing
    last_capture_time = 0
    capture_interval = 1.0  # seconds between automatic captures
    
    # For downsampling frames to improve performance
    frame_skip = 0
    frame_skip_rate = 2  # Process every Nth frame
    
    # Progress bar setup
    progress_bar_length = 30
    
    while captured < snapshots_count:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            break
            
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Skip frames for better performance
        frame_skip = (frame_skip + 1) % frame_skip_rate
        if frame_skip != 0:
            # Still show UI but skip face detection
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Enrolling: {username} ({captured}/{snapshots_count})", 
                      (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw progress bar
            progress = int(progress_bar_length * captured / snapshots_count)
            cv2.rectangle(frame, (20, 100), (20 + progress_bar_length*10, 130), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, 100), (20 + progress*10, 130), (0, 255, 0), -1)
            
            cv2.putText(frame, "Please look at the camera", 
                      (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("User Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on ESC
            if key == 27:  # ESC
                break
                
            continue
        
        # Process frame for face detection (only on non-skipped frames)
        # Use a smaller frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Use faster HOG model for enrollment
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        # Display enrollment UI
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Enrolling: {username} ({captured}/{snapshots_count})", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Please look at the camera", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw progress bar
        progress = int(progress_bar_length * captured / snapshots_count)
        cv2.rectangle(frame, (20, 100), (20 + progress_bar_length*10, 130), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 100), (20 + progress*10, 130), (0, 255, 0), -1)
        
        # Draw face rectangle if detected
        face_detected = False
        if face_locations:
            face_detected = True
            # Convert coordinates from small frame back to original frame
            for (top, right, bottom, left) in face_locations:
                # Scale coordinates back to original size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("User Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Auto-capture if face detected and enough time has passed
        current_time = time.time()
        if face_detected and (current_time - last_capture_time) > capture_interval:
            # Also allow manual capture with spacebar
            capture_now = True
        elif key == 32 and face_detected:  # Spacebar manual capture
            capture_now = True
        else:
            capture_now = False
            
        if capture_now:
            # Scale face locations back to original size for encoding
            scaled_locations = [(top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations]
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if face_encodings:
                # Save image
                img_path = os.path.join(user_folder, f"{username}_{captured+1}.jpg")
                cv2.imwrite(img_path, frame)
                captured += 1
                last_capture_time = current_time
                print(f"[INFO] Captured image {captured}/{snapshots_count}")
        
        # Exit on ESC
        if key == 27:  # ESC
            break
    
    cv2.destroyWindow("User Enrollment")
    print(f"[INFO] Enrollment complete for {username}")
    return

# --- Lock State Management ---
class LockStateManager:
    def __init__(self, cfg):
        self.locked = True
        self.consecutive_frames = cfg['face_recognition']['consecutive_frames']
        self.unlock_counter = 0
        self.lock_counter = 0
        self.current_user = None
        self.face_distance = 1.0
        # Grace period settings
        self.grace_period_s = cfg['face_recognition'].get('grace_period_s', 3.0)  # Default 3 seconds grace period
        self.last_face_time = 0
        self.in_grace_period = False
    
    def update(self, recognized_user, distance):
        now = time.time()
        
        if recognized_user:
            self.unlock_counter += 1
            self.lock_counter = 0
            self.current_user = recognized_user
            self.face_distance = distance
            self.last_face_time = now  # Update last time face was detected
            self.in_grace_period = False
        else:
            # Check if we're in grace period
            time_since_face = now - self.last_face_time
            if not self.locked and time_since_face < self.grace_period_s:
                # In grace period - don't increment lock counter
                self.in_grace_period = True
                # Reset unlock counter but don't increment lock counter
                self.unlock_counter = 0
            else:
                # Not in grace period or already locked
                self.lock_counter += 1
                self.unlock_counter = 0
                self.in_grace_period = False
        
        # Update lock state based on consecutive frames
        if self.unlock_counter >= self.consecutive_frames:
            self.locked = False
        if self.lock_counter >= self.consecutive_frames and not self.in_grace_period:
            self.locked = True
            self.current_user = None
        
        return self.locked, self.current_user, self.face_distance

def draw_ui(frame, state, fps, debug=False):
    """Optimized UI rendering for better performance"""
    h, w = frame.shape[:2]
    
    # Use a pre-allocated overlay for better performance
    # Only create overlay for the bottom info panel, not the whole frame
    bottom_overlay = np.zeros((120, 320, 3), dtype=np.uint8)
    cv2.rectangle(bottom_overlay, (0, 0), (320, 120), (0, 0, 0), -1)
    
    # Apply the overlay to just the bottom portion of the frame
    roi = frame[h-120:h-10, 10:330]
    if roi.shape[0] > 0 and roi.shape[1] > 0:  # Ensure ROI is valid
        cv2.addWeighted(bottom_overlay[:roi.shape[0], :roi.shape[1]], 0.5, roi, 0.5, 0, roi)
    
    # Use a more efficient font and reduce text rendering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if not debug else 0.8
    thickness = 1 if not debug else 2
    
    # Draw lock status banner at the top - only if needed
    if 'locked' in state:
        # Use rectangle directly instead of overlay for better performance
        if state['locked']:
            # Red banner for locked state
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
            cv2.putText(frame, "Access Denied", (w//2-100, 30), font, font_scale, (255, 255, 255), thickness)
        else:
            # Green banner for unlocked state with username
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 180, 0), -1)
            username = state.get('user', 'Unknown')
            cv2.putText(frame, f"{username} - Gestures Enabled", (w//2-150, 30), font, font_scale, (255, 255, 255), thickness)
        
        # Show face distance only in debug mode
        if debug and 'face_distance' in state:
            cv2.putText(frame, f"Face: {state['face_distance']:.3f}", 
                       (w-150, 70), font, 0.6, (255, 255, 255), 1)
    
    # Essential UI elements only
    cv2.putText(frame, f"Mode: {state.get('mode','')}", (20, h-90), font, font_scale, (255,255,255), thickness)
    
    # Only show gesture if there is one
    if state.get('gesture'):
        cv2.putText(frame, f"Gesture: {state.get('gesture','')}", (20, h-60), font, font_scale, (120,255,120), thickness)
    
    # Only show volume if needed
    if 'volume' in state:
        cv2.putText(frame, f"Vol: {state['volume']}%", (20, h-30), font, font_scale, (255,255,120), thickness)
    
    # Only show error if there is one
    if state.get('error'):
        cv2.putText(frame, f"ERROR: {state['error']}", (w//2-100, 70), font, font_scale, (0,0,255), thickness)
    
    # Always show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, h-30), font, font_scale, (255,255,255), thickness)
    
    return frame
