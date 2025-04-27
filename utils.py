#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py: Modular, robust gesture detection and smoothing for AirAuth.
All gesture detection is handled here, with FSM/debounce for reliability.
"""
import numpy as np
import cv2
import mediapipe as mp
import time
import subprocess
import pyautogui
import yaml
import os
import threading
from typing import Dict, List, Tuple, Optional
import face_recognition

def load_config(path):
    """Load configuration from YAML file with defaults"""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}. Using defaults.")
        # Return minimal default configuration
        return {
            'smoothing': {'method': 'one_euro', 'params': {'mincutoff': 0.8, 'beta': 0.004}},
            'detection': {'min_confidence': 0.85, 'min_tracking_confidence': 0.8},
            'gestures': {
                'left_click': {'hold_frames': 3, 'debounce_s': 0.3, 'threshold': 0.04},
                'right_click': {'hold_frames': 3, 'debounce_s': 0.3, 'threshold': 0.04},
                'scroll': {'thresh': 0.0025, 'dt_max': 0.4, 'sensitivity': 3.5},
                'tab_switch': {'hold_frames': 2, 'dt_max': 0.4, 'thresh': 0.025},
                'cooldown_s': 0.3
            },
            'camera': {
                'width': 1280,
                'height': 720,
                'fps': 30
            },
            'mac_keys': {
                'tab_forward': ["command", "tab"],
                'tab_back': ["command", "shift", "tab"],
                'zoom_in': ["command", "+"],
                'zoom_out': ["command", "-"]
            },
            'ui': {'show_hand_landmarks': True, 'show_fps': True},
            'face_recognition': {
                'model': "cnn",
                'tolerance': 0.55,
                'consecutive_frames': 3,
                'enroll_snapshots': 5,
                'folder_path': "faces/",
                'grace_period_s': 3.0
            },
            'drawing': {
                'line_thickness': 3,
                'line_color': [0, 0, 255],  # BGR
                'save_path': "~/Desktop/"
            }
        }

mp_hands = mp.solutions.hands

# --- Advanced Smoothing Filters ---
class OneEuroFilter:
    """
    Ultra-responsive 1€ filter for smooth cursor control.
    Reference: Casiez, G., Roussel, N., & Vogel, D. (2012).
    Revised for optimal gesture control responsiveness.
    """
    def __init__(self, freq=120.0, mincutoff=0.8, beta=0.004, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.last_time = None
        self._te = 1.0 / self.freq
        self._a_d = self._calculate_alpha(self.dcutoff)

    def _calculate_alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / self._te)

    def filter(self, x, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        if self.x_prev is None:
            self.last_time = now
            self.x_prev = x
            return x

        # Update based on actual frame rate
        dt = now - self.last_time
        if dt > 0:
            self.freq = 0.95 * self.freq + 0.05 * (1.0 / dt)
            self._te = 1.0 / self.freq
            self._a_d = self._calculate_alpha(self.dcutoff)

        # Calculate derivative
        dx = (x - self.x_prev) * self.freq

        # Filter derivative
        dx_hat = self._a_d * dx + (1 - self._a_d) * self.dx_prev

        # Adaptive cutoff based on movement speed
        movement_magnitude = abs(dx_hat)
        
        # Dynamic filtering adjustment
        if movement_magnitude > 3.0:
            # Fast movement - almost no filtering
            x_hat = x  # Direct passthrough
        elif movement_magnitude > 1.0:
            # Medium movement - light filtering
            cutoff = self.mincutoff * 3.0
            a = self._calculate_alpha(cutoff)
            x_hat = a * x + (1 - a) * self.x_prev
        else:
            # Slow movement - more filtering for precision
            cutoff = self.mincutoff + self.beta * movement_magnitude
            a = self._calculate_alpha(cutoff)
            x_hat = a * x + (1 - a) * self.x_prev

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.last_time = now

        return x_hat

    def apply(self, x):
        return self.filter(x)

# --- Robust Gesture State Machine ---
class GestureFSM:
    """
    Finite State Machine for gesture detection with debouncing and hold time
    """
    def __init__(self, hold_frames=3, reset_frames=3):
        self.frames_detected = 0
        self.frames_not_detected = 0
        self.active = False
        self.hold_frames = hold_frames
        self.reset_frames = reset_frames
        self.last_active_time = 0
    
    def update(self, detected, current_time=None):
        """Update FSM state based on detection results"""
        if current_time is None:
            current_time = time.time()
            
        if detected:
            self.frames_detected += 1
            self.frames_not_detected = 0
            
            if not self.active and self.frames_detected >= self.hold_frames:
                self.active = True
                self.last_active_time = current_time
                return True  # Newly activated
        else:
            self.frames_detected = 0
            self.frames_not_detected += 1
            
            if self.active and self.frames_not_detected >= self.reset_frames:
                self.active = False
                return False
        
        return None  # No state change

    def is_active(self):
        return self.active
        
    def get_duration(self, current_time=None):
        """Get duration for which gesture has been active"""
        if not self.active:
            return 0
        
        if current_time is None:
            current_time = time.time()
            
        return current_time - self.last_active_time

# --- Optimized Hand Analysis Functions ---
def get_finger_states(hand):
    """
    Determine which fingers are extended (1) or folded (0)
    Returns: [thumb, index, middle, ring, pinky]
    """
    if hand is None:
        return [0, 0, 0, 0, 0]
        
    lm = hand.landmark
    
    # Determine hand orientation (left/right hand)
    is_right_hand = lm[17].x > lm[5].x  # True for right hand, false for left hand
    
    # Different thumb folding detection based on hand type
    if is_right_hand:
        thumb_folded = lm[4].x > lm[3].x
    else:
        thumb_folded = lm[4].x < lm[3].x
    
    # For other fingers, check if tip is above (lower y value) than PIP joint
    index_extended = lm[8].y < lm[6].y
    middle_extended = lm[12].y < lm[10].y
    ring_extended = lm[16].y < lm[14].y
    pinky_extended = lm[20].y < lm[18].y
    
    return [
        1 if not thumb_folded else 0,
        1 if index_extended else 0,
        1 if middle_extended else 0,
        1 if ring_extended else 0,
        1 if pinky_extended else 0
    ]

def detect_pinch(hand, which='index', threshold=0.04):
    """
    Ultra-reliable pinch gesture detection with multi-point verification system
    
    Args:
        hand: MediaPipe hand landmarks
        which: 'index' or 'middle' finger
        threshold: Base threshold for pinch detection
        
    Returns:
        bool: True if pinch detected with high confidence
    """
    if hand is None:
        return False
    
    # Define landmark indices
    thumb_tip_idx = 4      # Thumb tip
    thumb_ip_idx = 3       # Thumb IP joint (second joint)
    thumb_mcp_idx = 2      # Thumb MCP joint (knuckle)
    
    # Get target finger indices based on which finger
    if which == 'index':
        finger_tip_idx = 8   # Index fingertip
        finger_pip_idx = 6   # Index PIP joint (middle joint)
        finger_mcp_idx = 5   # Index MCP joint (knuckle)
    elif which == 'middle':
        finger_tip_idx = 12  # Middle fingertip
        finger_pip_idx = 10  # Middle PIP joint
        finger_mcp_idx = 9   # Middle MCP joint
    else:
        return False
    
    # Get landmarks
    thumb_tip = hand.landmark[thumb_tip_idx]
    thumb_ip = hand.landmark[thumb_ip_idx]
    thumb_mcp = hand.landmark[thumb_mcp_idx]
    finger_tip = hand.landmark[finger_tip_idx]
    finger_pip = hand.landmark[finger_pip_idx]
    finger_mcp = hand.landmark[finger_mcp_idx]
    
    # Get finger states to verify proper hand pose
    fs = get_finger_states(hand)
    
    # PINCH DETECTION SYSTEM
    # =====================
    # 1. Primary check: 3D distance between thumb tip and finger tip
    dx = thumb_tip.x - finger_tip.x
    dy = thumb_tip.y - finger_tip.y
    dz = thumb_tip.z - finger_tip.z
    tip_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # 2. Secondary check: Proper finger extension based on which finger
    # Made more lenient by removing constraints on other fingers
    if which == 'index':
        # For index finger pinch: index should be somewhat extended
        proper_pose = (fs[1] == 1)  # Only check that index is extended
    else:  # Middle finger
        # For middle finger pinch: middle finger should be somewhat extended
        proper_pose = (fs[2] == 1)  # Only check that middle is extended
    
    # 3. Third check: Verify thumb is curled (not just extended)
    # Check angle between thumb segments to verify it's curling inward
    v1 = np.array([thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y, thumb_tip.z - thumb_ip.z])
    v2 = np.array([thumb_ip.x - thumb_mcp.x, thumb_ip.y - thumb_mcp.y, thumb_ip.z - thumb_mcp.z])
    v1_mag = np.sqrt(np.sum(v1*v1))
    v2_mag = np.sqrt(np.sum(v2*v2))
    
    if v1_mag * v2_mag == 0:
        thumb_angle = np.pi  # Invalid, default to 180 degrees
    else:
        dot_product = np.sum(v1 * v2) / (v1_mag * v2_mag)
        # Clamp dot product to valid range
        thumb_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Thumb should be bent for pinch (made more lenient, up to 170 degrees)
    thumb_curled = thumb_angle < (170 * np.pi / 180)
    
    # 4. Fourth check: Relative depth alignment - made more lenient
    # Thumb tip and finger tip should be at similar depths
    depth_aligned = abs(thumb_tip.z - finger_tip.z) < 0.08  # Increased threshold
    
    # 5. Fifth check: Relative distance assessment - made more lenient
    # Calculate distance between thumb tip and finger pip (middle joint)
    dx_tip_pip = thumb_tip.x - finger_pip.x
    dy_tip_pip = thumb_tip.y - finger_pip.y
    dz_tip_pip = thumb_tip.z - finger_pip.z
    tip_pip_distance = np.sqrt(dx_tip_pip*dx_tip_pip + dy_tip_pip*dy_tip_pip + dz_tip_pip*dz_tip_pip)
    
    # Tip-to-tip distance should be less than tip-to-PIP distance
    distance_check = tip_distance < (0.9 * tip_pip_distance)  # Made more lenient (was 0.7)
    
    # 6. Vertical alignment check - made more lenient
    vertical_aligned = abs(thumb_tip.y - finger_tip.y) < 0.08  # Increased threshold
    
    # Final decision system with confidence calculation
    # Each check contributes to overall confidence
    confidence = 0
    
    # Base distance check (most important - up to 45 points, increased from 40)
    if tip_distance < (threshold * 1.5):  # 50% larger threshold for more lenient detection
        # More weight on distance check and smoother curve
        confidence += 45 * (1 - min(tip_distance/(threshold*1.5), 1.0))
    
    # Additional checks (each up to 12 points, was 12 points each)
    if proper_pose:
        confidence += 11
    
    if thumb_curled:
        confidence += 11
    
    if depth_aligned:
        confidence += 11
    
    if distance_check:
        confidence += 11
    
    if vertical_aligned:
        confidence += 11
    
    # Return true only if confidence level is sufficient
    # LOWERED from 75 to 60 for more lenient detection
    return confidence >= 60

def detect_scroll(hand, prev_hand, thresh=0.0025):
    """
    Detect vertical scrolling gesture
    
    Args:
        hand: Current hand landmarks
        prev_hand: Previous frame hand landmarks
        thresh: Threshold for movement detection
        
    Returns:
        (bool, int): (detected, scroll_amount)
    """
    if hand is None or prev_hand is None:
        return False, 0
    
    fs = get_finger_states(hand)
    prev_fs = get_finger_states(prev_hand)
    
    # Detect open palm or L-shape (thumb + index)
    is_scroll_pose = (sum(fs) >= 3) or (fs[0] == 1 and fs[1] == 1 and sum(fs[2:]) == 0)
    was_scroll_pose = (sum(prev_fs) >= 3) or (prev_fs[0] == 1 and prev_fs[1] == 1 and sum(prev_fs[2:]) == 0)
    
    if is_scroll_pose and was_scroll_pose:
        # Track palm center for stability
        palm_y_now = hand.landmark[0].y
        palm_y_prev = prev_hand.landmark[0].y
        
        # Use wrist movement as backup
        wrist_y_now = hand.landmark[9].y
        wrist_y_prev = prev_hand.landmark[9].y
        
        # Weighted average for stability
        dy = 0.7 * (palm_y_now - palm_y_prev) + 0.3 * (wrist_y_now - wrist_y_prev)
        
        if abs(dy) > thresh:
            # Non-linear response curve for better control
            scroll_amount = int(np.sign(-dy) * (10 + 100 * pow(abs(dy), 1.5)))
            return True, scroll_amount
            
    return False, 0

def detect_tab_switch(hand, prev_hand, thresh=0.015):
    """
    Elite tab switching detection with confidence scoring, trajectory tracking and intent prediction
    
    Args:
        hand: Current hand landmarks
        prev_hand: Previous frame hand landmarks
        thresh: Base threshold for horizontal movement
        
    Returns:
        (bool, bool): (detected, direction) - direction True for right, False for left
    """
    if hand is None or prev_hand is None:
        return False, False
    
    # Step 1: Check for proper hand pose (peace sign with only index and middle extended)
    fs = get_finger_states(hand)
    prev_fs = get_finger_states(prev_hand)
    
    # Must have index and middle fingers extended, others folded
    is_peace_sign = (fs[1] == 1 and fs[2] == 1 and fs[0] == 0 and fs[3] == 0 and fs[4] == 0)
    was_peace_sign = (prev_fs[1] == 1 and prev_fs[2] == 1 and prev_fs[0] == 0 and prev_fs[3] == 0 and prev_fs[4] == 0)
    
    if not (is_peace_sign and was_peace_sign):
        return False, False
    
    # Step 2: Multi-point verification system uses 6 key points
    # This creates redundancy and reduces false positives significantly
    points = [
        (hand.landmark[8], prev_hand.landmark[8]),     # Index fingertip
        (hand.landmark[12], prev_hand.landmark[12]),   # Middle fingertip
        (hand.landmark[6], prev_hand.landmark[6]),     # Index PIP joint
        (hand.landmark[10], prev_hand.landmark[10]),   # Middle PIP joint
        (hand.landmark[5], prev_hand.landmark[5]),     # Index MCP joint
        (hand.landmark[9], prev_hand.landmark[9])      # Middle MCP joint
    ]
    
    # Step 3: Calculate movement vectors and analyze
    dx_values = []  # Horizontal movement of each point
    dy_values = []  # Vertical movement of each point
    
    for curr, prev in points:
        dx_values.append(curr.x - prev.x)
        dy_values.append(curr.y - prev.y)
    
    # Step 4: Check directional consistency
    # All points should be moving in the same direction horizontally
    dx_sign_positive = sum(1 for dx in dx_values if dx > 0)
    dx_sign_negative = sum(1 for dx in dx_values if dx < 0)
    
    # If points are split in different directions, reject the gesture
    if dx_sign_positive > 0 and dx_sign_negative > 0:
        # Mixed directions detected - not a clean swipe
        return False, False
    
    # Step 5: Calculate magnitude of motion
    avg_dx = sum(dx_values) / len(dx_values)
    avg_dy = sum(abs(dy_val) for dy_val in dy_values) / len(dy_values)
    
    # Step 6: Apply robust decision criteria
    # 1. Horizontal movement must exceed threshold
    # 2. Horizontal movement must be significantly larger than vertical (clean swipe)
    # 3. Movement must have consistent direction across all tracked points
    if (abs(avg_dx) > thresh and          # Sufficient movement
        abs(avg_dx) > avg_dy * 2.0 and    # More horizontal than vertical
        (dx_sign_positive == len(points) or dx_sign_negative == len(points))):  # Consistent direction
        
        # Calculate intent confidence score (0-100)
        confidence = min(100, 
                         (abs(avg_dx) / thresh) * 50 +  # Movement magnitude 
                         (abs(avg_dx) / (avg_dy + 0.001)) * 25 +  # Horizontal vs vertical ratio
                         min(25, (max(dx_sign_positive, dx_sign_negative) / len(points)) * 25))  # Direction consistency
        
        # Only trigger with high confidence
        if confidence > 75:
            # Direction: negative dx = left swipe, positive = right swipe
            # Return: True = next tab (left swipe), False = previous tab (right swipe)
            return True, avg_dx < 0
    
    return False, False

def detect_victory(hand, min_dist=0.07, min_angle=15):
    """
    Detect victory sign (V) for drawing mode toggle
    
    Args:
        hand: MediaPipe hand landmarks
        min_dist: Minimum normalized distance between fingertips
        min_angle: Minimum angle (degrees) between fingers
        
    Returns:
        bool: True if victory sign detected
    """
    if hand is None:
        return False
        
    fs = get_finger_states(hand)
    
    # Victory sign requires index and middle extended, others folded
    if not (fs[1] == 1 and fs[2] == 1 and fs[0] == 0 and fs[3] == 0 and fs[4] == 0):
        return False
        
    lm = hand.landmark
    
    # Check that fingertips are sufficiently apart
    tip_dist = np.sqrt((lm[8].x - lm[12].x)**2 + (lm[8].y - lm[12].y)**2)
    if tip_dist < min_dist:
        return False
        
    # Calculate vectors for each finger
    v1 = np.array([lm[8].x - lm[5].x, lm[8].y - lm[5].y])
    v2 = np.array([lm[12].x - lm[9].x, lm[12].y - lm[9].y])
    
    # Calculate angle between vectors
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:  # Avoid division by zero
        return False
        
    angle = np.degrees(np.arccos(np.clip(dot/norm, -1.0, 1.0)))
    
    return angle > min_angle

def detect_fist(hand):
    """
    Detect closed fist pose
    
    Args:
        hand: MediaPipe hand landmarks
        
    Returns:
        bool: True if fist detected
    """
    if hand is None:
        return False
        
    fs = get_finger_states(hand)
    return sum(fs) == 0  # All fingers closed

def detect_neutral(hand):
    """
    Detect neutral pose (open hand)
    
    Args:
        hand: MediaPipe hand landmarks
        
    Returns:
        bool: True if neutral pose detected
    """
    if hand is None:
        return False
        
    fs = get_finger_states(hand)
    return sum(fs) == 5  # All fingers extended

def detect_zoom(hand1, hand2, prev_hand1, prev_hand2, thresh=0.06):
    """
    Detect zoom gesture using two hands
    
    Args:
        hand1, hand2: Current hand landmarks
        prev_hand1, prev_hand2: Previous hand landmarks
        thresh: Movement threshold
        
    Returns:
        (bool, bool): (detected, zoom_in)
    """
    if hand1 is None or hand2 is None or prev_hand1 is None or prev_hand2 is None:
        return False, False
        
    # Use index fingertips distance to determine zoom
    dist_now = np.sqrt(
        (hand1.landmark[8].x - hand2.landmark[8].x)**2 + 
        (hand1.landmark[8].y - hand2.landmark[8].y)**2
    )
    
    dist_prev = np.sqrt(
        (prev_hand1.landmark[8].x - prev_hand2.landmark[8].x)**2 + 
        (prev_hand1.landmark[8].y - prev_hand2.landmark[8].y)**2
    )
    
    # Check for significant distance change
    if abs(dist_now - dist_prev) > thresh:
        return True, dist_now > dist_prev  # True = zoom in, False = zoom out
        
    return False, False

def detect_zoom_gesture(hand1, hand2, prev_hand1, prev_hand2, threshold=0.04):
    """
    Detect zoom gesture using both hands in pinch position
    
    Args:
        hand1, hand2: Current hand landmarks for both hands
        prev_hand1, prev_hand2: Previous hand landmarks for both hands
        threshold: Minimum change in distance to trigger zoom
        
    Returns:
        (detected, zoom_in, magnitude): Whether detected, direction, and strength
    """
    if hand1 is None or hand2 is None or prev_hand1 is None or prev_hand2 is None:
        return False, False, 0
    
    # Check if both hands are in pinch position (thumb and index finger close)
    h1_pinch = detect_pinch(hand1, which='index', threshold=0.06)
    h2_pinch = detect_pinch(hand2, which='index', threshold=0.06)
    
    if not (h1_pinch and h2_pinch):
        return False, False, 0
    
    # Calculate distance between pinch points
    h1_pinch_point = (hand1.landmark[4].x, hand1.landmark[4].y)  # thumb tip
    h2_pinch_point = (hand2.landmark[4].x, hand2.landmark[4].y)  # thumb tip
    
    prev_h1_pinch_point = (prev_hand1.landmark[4].x, prev_hand1.landmark[4].y)
    prev_h2_pinch_point = (prev_hand2.landmark[4].x, prev_hand2.landmark[4].y)
    
    # Current distance between pinch points
    current_dist = np.sqrt(
        (h1_pinch_point[0] - h2_pinch_point[0])**2 + 
        (h1_pinch_point[1] - h2_pinch_point[1])**2
    )
    
    # Previous distance between pinch points
    prev_dist = np.sqrt(
        (prev_h1_pinch_point[0] - prev_h2_pinch_point[0])**2 + 
        (prev_h1_pinch_point[1] - prev_h2_pinch_point[1])**2
    )
    
    # Change in distance
    delta = current_dist - prev_dist
    
    # Check if change is significant enough
    if abs(delta) > threshold:
        # Determine zoom direction and strength
        zoom_in = delta > 0  # Hands moving apart = zoom in, hands moving closer = zoom out
        magnitude = min(abs(delta) * 5, 5)  # Limit maximum zoom strength
        return True, zoom_in, magnitude
    
    return False, False, 0

def toggle_media_playback():
    """Toggle media playback on macOS"""
    try:
        subprocess.run(
            ['osascript', '-e', 'tell application "System Events" to key code 100'], 
            check=True
        )
        return True
    except Exception as e:
        print(f"[WARN] Media toggle failed: {e}")
        return False

def adjust_volume(direction, step=2):
    """Adjust system volume"""
    try:
        # Get current volume
        get_vol = subprocess.run(
            ['osascript', '-e', 'output volume of (get volume settings)'],
            capture_output=True, text=True
        )
        vol = int(get_vol.stdout.strip())
        
        # Calculate new volume
        new_vol = max(0, min(100, vol + (step if direction=='up' else -step)))
        
        # Set new volume
        subprocess.run(['osascript', '-e', f'set volume output volume {new_vol}'])
        return new_vol
    except Exception as e:
        print(f"[WARN] Volume adjust failed: {e}")
        return None

# --- Face Recognition Functions ---
# Cache for face encodings to avoid redundant calculations
face_encoding_cache = {}

def load_face_data(folder_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Load face recognition data from the filesystem
    
    Args:
        folder_path: Path to face data folder
        
    Returns:
        Dict mapping usernames to lists of face encodings
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
            known_embeddings = face_state.known_embeddings.copy()
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
try:
    face_thread = threading.Thread(target=face_recognition_thread, daemon=True)
    face_thread.start()
except Exception as e:
    print(f"[ERROR] Failed to start face recognition thread: {e}")

def recognize_face(frame: np.ndarray, known_embeddings: Dict[str, List[np.ndarray]], cfg) -> Tuple[Optional[str], float]:
    """Thread-safe wrapper for face recognition"""
    with face_state.lock:
        # Skip frames to improve performance
        face_state.frame_count = (face_state.frame_count + 1) % (face_state.skip_frames + 1)
        if face_state.frame_count != 0:
            return face_state.result

        # If not currently processing, submit a new frame
        if not face_state.processing and face_state.frame_to_process is None:
            face_state.frame_to_process = frame.copy()
            face_state.known_embeddings = known_embeddings.copy()
            face_state.cfg = cfg

        return face_state.result

def recognize_face_internal(frame: np.ndarray, known_embeddings: Dict[str, List[np.ndarray]], cfg) -> Tuple[Optional[str], float]:
    """
    Process face recognition
    
    Args:
        frame: Video frame to analyze
        known_embeddings: Dict of known face encodings
        cfg: Configuration
        
    Returns:
        (username, confidence): User ID and confidence score
    """
    # No known faces to compare against
    if not known_embeddings:
        return None, 1.0

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 25% of original size
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find faces in the current frame
    model = cfg['face_recognition'].get('model', 'cnn')
    face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
    if not face_locations:
        return None, 1.0
        
    # If multiple faces detected, use the largest face (closest to camera)
    if len(face_locations) > 1:
        largest_area = 0
        largest_face_idx = 0
        
        for i, (top, right, bottom, left) in enumerate(face_locations):
            area = (bottom - top) * (right - left)
            if area > largest_area:
                largest_area = area
                largest_face_idx = i
                
        # Only use the largest face
        face_locations = [face_locations[largest_face_idx]]

    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=2)
    if not face_encodings:
        return None, 1.0

    # Compare with known faces
    match_scores = {}
    min_distance = float('inf')
    best_match = None

    for username, stored_encodings in known_embeddings.items():
        # Calculate average distance across all encodings for this user
        total_distance = 0
        match_count = 0
        
        for stored_encoding in stored_encodings:
            face_distance = face_recognition.face_distance([stored_encoding], face_encodings[0])[0]
            
            # Only count distances below a certain threshold
            if face_distance < 0.6:  # Pre-filtering threshold
                total_distance += face_distance
                match_count += 1
        
        # Calculate average distance if we have matches
        if match_count > 0:
            avg_distance = total_distance / match_count
            match_scores[username] = avg_distance
            
            # Track best match
            if avg_distance < min_distance:
                min_distance = avg_distance
                best_match = username

    # Return match if distance is below tolerance
    tolerance = cfg['face_recognition']['tolerance']
    if min_distance <= tolerance and best_match is not None:
        return best_match, min_distance
    else:
        return None, min_distance if min_distance < float('inf') else 1.0

class LockStateManager:
    """
    Manages authentication state based on face recognition
    """
    def __init__(self, cfg):
        self.locked = True
        self.consecutive_frames = cfg['face_recognition']['consecutive_frames']
        self.unlock_counter = 0
        self.lock_counter = 0
        self.current_user = None
        self.face_distance = 1.0
        
        # Grace period settings
        self.grace_period_s = cfg['face_recognition'].get('grace_period_s', 3.0)
        self.last_face_time = 0
        self.in_grace_period = False

    def update(self, recognized_user, distance):
        """
        Update lock state based on recognition results
        
        Args:
            recognized_user: Detected username or None
            distance: Face distance (lower = better match)
            
        Returns:
            (locked, username, distance): Current state and metrics
        """
        now = time.time()

        if recognized_user:
            self.unlock_counter += 1
            self.lock_counter = 0
            self.current_user = recognized_user
            self.face_distance = distance
            self.last_face_time = now
            self.in_grace_period = False
        else:
            # Check if we're in grace period
            time_since_face = now - self.last_face_time
            if not self.locked and time_since_face < self.grace_period_s:
                # In grace period - don't increment lock counter
                self.in_grace_period = True
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

def enroll_user(username: str, cap: cv2.VideoCapture, cfg) -> None:
    """
    Enroll a new user for face recognition
    
    Args:
        username: User ID to enroll
        cap: OpenCV video capture object
        cfg: Configuration
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
    capture_interval = 1.0  # Seconds between automatic captures

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

        # Process frame for face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")

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
            capture_now = True
        elif key == 32 and face_detected:  # Spacebar manual capture
            capture_now = True
        else:
            capture_now = False

        if capture_now:
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

def draw_ui(frame, state, fps, debug=False):
    """
    Draw UI overlay on the frame
    
    Args:
        frame: Video frame to draw on
        state: Current UI state dictionary
        fps: Current FPS
        debug: Whether to show extra debug info
        
    Returns:
        frame with UI elements drawn
    """
    h, w = frame.shape[:2]

    # Create bottom info panel overlay
    bottom_height = 120
    bottom_width = 320
    overlay = np.zeros((bottom_height, bottom_width, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, 0), (bottom_width, bottom_height), (0, 0, 0), -1)

    # Apply overlay to bottom left of frame
    y_start = h - bottom_height - 10
    y_end = y_start + overlay.shape[0]
    x_start = 10
    x_end = x_start + overlay.shape[1]
    
    if y_start >= 0 and x_start >= 0 and y_end <= h and x_end <= w:
        roi = frame[y_start:y_end, x_start:x_end]
        cv2.addWeighted(overlay, 0.7, roi, 0.3, 0, roi)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1 if not debug else 2
    text_color = (255, 255, 255)

    # Draw lock status banner at the top
    if 'locked' in state:
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

    # Show mode
    cv2.putText(frame, f"Mode: {state.get('mode','')}", (20, h-90), font, font_scale, text_color, thickness)

    # Show current gesture
    if state.get('gesture'):
        cv2.putText(frame, f"Gesture: {state['gesture']}", (20, h-60), font, font_scale, (120, 255, 120), thickness)

    # Show volume if relevant
    if 'volume' in state:
        cv2.putText(frame, f"Vol: {state['volume']}%", (20, h-30), font, font_scale, (255, 255, 120), thickness)

    # Show errors
    if state.get('error'):
        cv2.putText(frame, f"ERROR: {state['error']}", (w//2-100, 70), font, font_scale, (0, 0, 255), thickness)

    # Always show FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, h-30), font, font_scale, text_color, thickness)

    return frame
