#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirAuth: Advanced gesture control system with face authentication
A macOS-compatible solution for controlling your computer with hand gestures.
"""

import os
import sys
import platform
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import yaml
import pyautogui
import json
import time
import argparse
from utils import *
from collections import deque
import face_recognition

# --- Enforce conda environment ---
if os.environ.get('CONDA_DEFAULT_ENV') != 'vis':
    sys.stderr.write("[ERROR] Please activate the 'vis' conda environment before running this script.\nUse: conda activate vis\n")
    sys.exit(1)

# --- Setup for smooth operation ---
pyautogui.PAUSE = 0.01  # Faster PyAutoGUI commands
pyautogui.FAILSAFE = True  # Enable failsafe corner

# --- macOS Permissions Check ---
if platform.system() == 'Darwin':
    # Accessibility check (mouse/keyboard control)
    try:
        import Quartz
        if not Quartz.CGPreflightListenEventAccess():
            print("[ERROR] Accessibility permissions not granted.")
            print("Go to System Settings > Privacy & Security > Accessibility, and add Terminal or your Python interpreter.")
            sys.exit(1)
    except ImportError:
        print("[WARNING] Quartz not installed; skipping Accessibility check.")
    
    # Camera check
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera access denied or unavailable.")
        print("Go to System Settings > Privacy & Security > Camera, and allow access for Terminal or your Python interpreter.")
        sys.exit(1)
    cap.release()

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description='AirAuth: Advanced Gesture Control System')
parser.add_argument('--config', default='config.yaml', help='Path to config file')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with extra visuals and logging')
parser.add_argument('--camera', type=int, default=0, help='Camera index')
parser.add_argument('--enroll', type=str, help='Enroll a new user with the specified username')
parser.add_argument('--no-face-auth', action='store_true', help='Skip face authentication')
args = parser.parse_args()

# --- Configuration Loading ---
def load_config_safe(path):
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"[INFO] Configuration loaded from {path}")
            return config
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}. Using default configuration.")
        return load_config(None)  # Load default config from utils.py

config = load_config_safe(args.config)
debug = args.debug

def debug_print(msg):
    """Print debug messages only in debug mode"""
    if debug:
        print(f"[DEBUG] {msg}")

# --- Screen Setup ---
screen_width, screen_height = pyautogui.size()
print(f"[INFO] Screen resolution: {screen_width}x{screen_height}")

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure MediaPipe for optimal performance
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=config['detection']['min_confidence'],
    min_tracking_confidence=config['detection'].get('min_tracking_confidence', 0.7),
    model_complexity=1  # 1 gives a good balance between accuracy and speed
)

# --- Face Recognition Setup ---
if not args.no_face_auth:
    face_data = load_face_data(config['face_recognition']['folder_path'])
    lock_manager = LockStateManager(config)
    
    if not face_data:
        print("[WARNING] No face data found. You may want to enroll a user first.")
        print("          Use the --enroll flag: python app.py --enroll <username>")
        print("          Running with authentication disabled.")
        args.no_face_auth = True

# --- Application State ---
class AppState:
    """Class to manage application state"""
    def __init__(self):
        # General application state
        self.mode = 'CURSOR'  # Modes: CURSOR, DRAWING
        self.exiting = False
        self.fps = 0
        self.frame_time = 0
        self.prev_frame_time = 0
        
        # Authentication state
        self.locked = not args.no_face_auth
        self.current_user = None
        self.face_distance = 1.0
        
        # Hand tracking state
        self.prev_hand = None
        self.hand_present = False
        self.hand_landmarks = None
        self.prev_hand_landmarks = None
        
        # Cursor state - optimized filter parameters for minimal lag
        self.cursor_filter_x = OneEuroFilter(freq=144.0, mincutoff=0.001, beta=0.001)
        self.cursor_filter_y = OneEuroFilter(freq=144.0, mincutoff=0.001, beta=0.001)
        
        # Gesture state
        self.last_gesture_time = 0
        self.gesture_cooldown = config['gestures']['cooldown_s']
        self.current_gesture = None
        self.finger_states = [0, 0, 0, 0, 0]
        
        # FSMs for gesture detection - more responsive settings
        self.pinch_fsm = GestureFSM(
            hold_frames=config['gestures']['left_click']['hold_frames'], 
            reset_frames=1  # Faster reset
        )
        self.rpinch_fsm = GestureFSM(
            hold_frames=config['gestures']['right_click']['hold_frames'], 
            reset_frames=1  # Faster reset
        )
        self.scroll_fsm = GestureFSM(hold_frames=1, reset_frames=1) # More responsive
        self.tab_fsm = GestureFSM(
            hold_frames=config['gestures']['tab_switch']['hold_frames'], 
            reset_frames=1
        )
        self.victory_fsm = GestureFSM(hold_frames=3, reset_frames=2) # More responsive
        self.fist_fsm = GestureFSM(hold_frames=2, reset_frames=2) # More responsive
        
        # Drawing mode state
        self.drawing_canvas = None
        self.last_draw_point = None
        
        # UI state for display
        self.ui_state = {
            'mode': 'Cursor',
            'gesture': None,
            'locked': self.locked,
            'user': None,
            'error': None,
            'show_fps': True
        }
    
    def update_locked_state(self, locked, user=None, face_distance=1.0):
        """Update authentication state"""
        self.locked = locked
        self.current_user = user
        self.face_distance = face_distance
        self.ui_state['locked'] = locked
        self.ui_state['user'] = user
        self.ui_state['face_distance'] = face_distance
    
    def update_mode(self, new_mode):
        """Update application mode"""
        if new_mode != self.mode:
            self.mode = new_mode
            self.ui_state['mode'] = new_mode.capitalize()
            if new_mode == 'DRAWING':
                # Reset drawing canvas when entering drawing mode
                self.drawing_canvas = None
                self.last_draw_point = None
                debug_print("Drawing mode activated")
            else:
                debug_print("Cursor mode activated")
            return True
        return False
    
    def update_gesture(self, gesture_name):
        """Update current gesture display"""
        if gesture_name != self.current_gesture:
            self.current_gesture = gesture_name
            self.ui_state['gesture'] = gesture_name
            return True
        return False

    def reset_gesture(self):
        """Reset current gesture"""
        self.current_gesture = None
        self.ui_state['gesture'] = None

    def update_fps(self, now):
        """Update FPS calculation"""
        self.frame_time = now
        time_diff = self.frame_time - self.prev_frame_time
        
        if time_diff > 0:
            # Exponential moving average for smoother FPS display
            instantaneous_fps = 1.0 / time_diff
            alpha = 0.1  # Lower alpha = more smoothing
            self.fps = alpha * instantaneous_fps + (1 - alpha) * self.fps if self.fps > 0 else instantaneous_fps
        
        self.prev_frame_time = self.frame_time
        self.ui_state['fps'] = self.fps

    def can_trigger_gesture(self, cooldown=None):
        """Check if we can trigger a new gesture based on cooldown"""
        if cooldown is None:
            cooldown = self.gesture_cooldown
            
        now = time.time()
        if now - self.last_gesture_time > cooldown:
            self.last_gesture_time = now
            return True
        return False

    def update_finger_states(self, hand_landmarks):
        """Update finger states based on hand landmarks"""
        if hand_landmarks:
            self.finger_states = get_finger_states(hand_landmarks)
        else:
            self.finger_states = [0, 0, 0, 0, 0]

# Initialize application state
state = AppState()

# --- Camera Setup ---
def setup_camera():
    """Setup camera with desired properties"""
    camera_index = args.camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera {camera_index}")
        sys.exit(1)
    
    # Configure camera for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, config['camera']['fps'])
    
    # Additional optimizations
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer for reduced latency
    
    # Check the actual properties (may differ from requested)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    return cap

# --- Gesture Processing Functions ---
def process_hand_landmarks(frame, process_every=1):
    """
    Process frame to detect hand landmarks using MediaPipe
    
    Args:
        frame: Input video frame
        process_every: Process every Nth frame for performance
        
    Returns:
        hand_landmarks: List of detected hand landmarks
    """
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Initialize landmarks list
    detected_landmarks = [None, None]
    
    # Extract landmarks if hands are detected
    if results.multi_hand_landmarks:
        # Get up to 2 hands
        for i, landmarks in enumerate(results.multi_hand_landmarks[:2]):
            detected_landmarks[i] = landmarks
            
            # Draw landmarks in debug mode
            if debug and config['ui']['show_hand_landmarks']:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )
    
    return detected_landmarks

def process_cursor_movement(hand_landmarks):
    """
    Elite cursor control system with predictive movement and adaptive stabilization
    
    Args:
        hand_landmarks: Hand landmarks from MediaPipe
        
    Returns:
        cursor_pos: (x, y) cursor position
    """
    if hand_landmarks is None:
        return None
    
    # Multi-point tracking system for robust cursor control
    index_tip = hand_landmarks.landmark[8]    # Primary control point
    middle_tip = hand_landmarks.landmark[12]  # Secondary reference
    index_pip = hand_landmarks.landmark[6]    # Index PIP joint for stability
    index_mcp = hand_landmarks.landmark[5]    # Index MCP joint for stability
    wrist = hand_landmarks.landmark[0]        # Wrist anchor point
    
    # Get finger pose to determine intended control mode
    fs = get_finger_states(hand_landmarks)
    
    # Three control modes for different precision needs:
    # 1. Precision mode: Only index finger extended
    # 2. Balanced mode: Index + thumb (pointing)
    # 3. Normal mode: Other configurations
    precision_mode = fs[1] == 1 and fs[0] == 0 and fs[2] == 0 and fs[3] == 0 and fs[4] == 0
    balanced_mode = fs[1] == 1 and fs[0] == 1 and fs[2] == 0 and fs[3] == 0 and fs[4] == 0
    
    # Dynamic point weighting based on control mode
    if precision_mode:
        # Precision mode: Heavy focus on fingertip with slight stabilization
        # Best for hitting small targets like buttons precisely
        raw_x = index_tip.x * 0.9 + index_pip.x * 0.07 + wrist.x * 0.03
        raw_y = index_tip.y * 0.9 + index_pip.y * 0.07 + wrist.y * 0.03
        # Configure filter for precision
        state.cursor_filter_x.mincutoff = 0.04  # More filtering
        state.cursor_filter_y.mincutoff = 0.04
        state.cursor_filter_x.beta = 0.0015     # Moderate smoothness
        state.cursor_filter_y.beta = 0.0015
        dead_zone = 2  # Smaller dead zone for precision
    elif balanced_mode:
        # Balanced mode: Good blend of precision and stability
        # Best for normal navigation and most tasks
        raw_x = index_tip.x * 0.8 + index_pip.x * 0.1 + index_mcp.x * 0.06 + wrist.x * 0.04
        raw_y = index_tip.y * 0.8 + index_pip.y * 0.1 + index_mcp.y * 0.06 + wrist.y * 0.04
        # Configure filter for balanced control
        state.cursor_filter_x.mincutoff = 0.06  # Balanced filtering
        state.cursor_filter_y.mincutoff = 0.06
        state.cursor_filter_x.beta = 0.002      # Smooth but responsive
        state.cursor_filter_y.beta = 0.002
        dead_zone = 3  # Medium dead zone
    else:
        # Normal mode: Stability over precision
        # Best for browsing and casual navigation
        raw_x = index_tip.x * 0.65 + middle_tip.x * 0.05 + index_pip.x * 0.15 + index_mcp.x * 0.1 + wrist.x * 0.05
        raw_y = index_tip.y * 0.65 + middle_tip.y * 0.05 + index_pip.y * 0.15 + index_mcp.y * 0.1 + wrist.y * 0.05
        # Configure filter for stability
        state.cursor_filter_x.mincutoff = 0.08  # More filtering for stability
        state.cursor_filter_y.mincutoff = 0.08
        state.cursor_filter_x.beta = 0.003      # More smoothness
        state.cursor_filter_y.beta = 0.003
        dead_zone = 3  # Standard dead zone
    
    # Apply optimized filtering with updated parameters
    filtered_x = state.cursor_filter_x.apply(raw_x)
    filtered_y = state.cursor_filter_y.apply(raw_y)
    
    # Apply non-linear mapping for screen traversal efficiency
    # This creates a slight acceleration effect for traversing large screen distances
    def non_linear_map(val):
        # Center-weighted sigmoid-like mapping
        # Makes it easier to go edge-to-edge while maintaining precision in the center
        center_offset = val - 0.5
        acceleration = 1.0 + min(0.6, abs(center_offset) * 1.2)
        return 0.5 + center_offset * acceleration
    
    # Apply non-linear mapping conditionally - only for larger movements
    # This keeps small precision movements 1:1 while accelerating larger movements
    curr_x, curr_y = pyautogui.position()
    norm_curr_x, norm_curr_y = curr_x / screen_width, curr_y / screen_height
    
    # Calculate distance between filtered and current normalized positions
    dist_x = abs(filtered_x - norm_curr_x)
    dist_y = abs(filtered_y - norm_curr_y)
    
    # Only apply non-linear mapping for movements above threshold
    if dist_x > 0.05 or dist_y > 0.05:
        mapped_x = non_linear_map(filtered_x)
        mapped_y = non_linear_map(filtered_y)
    else:
        mapped_x = filtered_x
        mapped_y = filtered_y
    
    # Map to screen coordinates
    cursor_x = int(mapped_x * screen_width)
    cursor_y = int(mapped_y * screen_height)
    
    # Get current cursor position for distance calculation
    current_x, current_y = pyautogui.position()
    
    # Calculate movement distance for adaptive smoothing
    distance = np.sqrt((cursor_x - current_x)**2 + (cursor_y - current_y)**2)
    
    # Only move if beyond dead zone threshold
    if distance > dead_zone:
        # Adaptive movement smoothing based on speed and distance
        # Fast for big movements, smooth for small adjustments
        speed_factor = min(distance / 100, 1.0)  # 0.0-1.0 based on distance
        
        # Dynamic smoothing factor: more aggressive for larger movements
        # Creates a feeling of momentum without the lag of interpolation
        smoothing_factor = 0.25 + speed_factor * 0.45  # Range: 0.25-0.7
        
        # Apply smoothing between current and target positions
        new_x = int(current_x + (cursor_x - current_x) * smoothing_factor)
        new_y = int(current_y + (cursor_y - current_y) * smoothing_factor)
        
        # Secondary stability filter to combat jitter during micro-movements
        if distance < 10:  # For very small movements
            # Apply additional temporal smoothing
            if hasattr(state, 'prev_cursor_pos') and state.prev_cursor_pos is not None:
                prev_x, prev_y = state.prev_cursor_pos
                new_x = int(new_x * 0.7 + prev_x * 0.3)
                new_y = int(new_y * 0.7 + prev_y * 0.3)
        
        # Ensure cursor stays within screen boundaries
        new_x = max(0, min(new_x, screen_width - 1))
        new_y = max(0, min(new_y, screen_height - 1))
        
        # Store position for next frame's reference
        state.prev_cursor_pos = (new_x, new_y)
        
        # Move cursor with zero duration for minimal lag
        pyautogui.moveTo(new_x, new_y, duration=0)
        return (new_x, new_y)
    else:
        # Store current position
        state.prev_cursor_pos = (current_x, current_y)
        return (current_x, current_y)

def process_clicks(hand_landmarks):
    """
    Enhanced click detection with dual-verification system
    
    Args:
        hand_landmarks: Hand landmarks from MediaPipe
        
    Returns:
        (left_click, right_click): Whether left/right clicks were triggered
    """
    left_click = False
    right_click = False
    
    if hand_landmarks is None:
        state.pinch_fsm.update(False)
        state.rpinch_fsm.update(False)
        return left_click, right_click
    
    # PRIMARY DETECTION METHOD: Pinch Detection
    # Left click (index finger pinch)
    is_left_pinch = detect_pinch(
        hand_landmarks, 
        which='index',
        threshold=config['gestures']['left_click']['threshold']
    )
    
    # SECONDARY DETECTION METHOD: Distance-based fallback
    # This simpler method acts as a backup if the primary method fails
    # It only checks the distance between thumb tip and index tip
    if not is_left_pinch:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate simple 3D distance
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        dz = thumb_tip.z - index_tip.z
        simple_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Use a more permissive threshold for the fallback
        fallback_threshold = config['gestures']['left_click']['threshold'] * 1.2
        if simple_distance < fallback_threshold:
            is_left_pinch = True
    
    # Process with finite state machine for robustness
    pinch_updated = state.pinch_fsm.update(is_left_pinch)
    
    if pinch_updated is True:  # Just activated
        if state.can_trigger_gesture(config['gestures']['left_click']['debounce_s']):
            pyautogui.click(button='left')
            state.update_gesture('Left Click')
            left_click = True
            debug_print("Left click triggered")
    
    # RIGHT CLICK DETECTION - Same dual approach
    # Primary detection
    is_right_pinch = detect_pinch(
        hand_landmarks,
        which='middle',
        threshold=config['gestures']['right_click']['threshold']
    )
    
    # Secondary detection fallback
    if not is_right_pinch:
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        dx = thumb_tip.x - middle_tip.x
        dy = thumb_tip.y - middle_tip.y
        dz = thumb_tip.z - middle_tip.z
        simple_distance = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        fallback_threshold = config['gestures']['right_click']['threshold'] * 1.2
        if simple_distance < fallback_threshold:
            is_right_pinch = True
    
    rpinch_updated = state.rpinch_fsm.update(is_right_pinch)
    
    if rpinch_updated is True:  # Just activated
        if state.can_trigger_gesture(config['gestures']['right_click']['debounce_s']):
            pyautogui.click(button='right')
            state.update_gesture('Right Click')
            right_click = True
            debug_print("Right click triggered")
    
    return left_click, right_click

def process_tab_switch(hand_landmarks, prev_hand_landmarks):
    """
    Enhanced tab switching with gesture prediction and controlled speed 
    to prevent multiple tab changes at once
    
    Args:
        hand_landmarks: Current hand landmarks
        prev_hand_landmarks: Previous hand landmarks
        
    Returns:
        switched_tab: Whether tab switching occurred
    """
    switched_tab = False
    
    if hand_landmarks is None or prev_hand_landmarks is None:
        state.tab_fsm.update(False)
        return switched_tab
    
    # Use strict finger state detection for tab switching gesture (peace sign)
    fs = state.finger_states
    is_peace_sign = (fs[1] == 1 and fs[2] == 1 and fs[0] == 0 and fs[3] == 0 and fs[4] == 0)
    
    # Track if we're potentially in a tab switching motion
    if is_peace_sign:
        # Detect tab switch gesture
        tab_detected, tab_direction = detect_tab_switch(
            hand_landmarks,
            prev_hand_landmarks,
            thresh=config['gestures']['tab_switch'].get('thresh', 0.015)
        )
        
        # Update FSM for gesture tracking
        tab_updated = state.tab_fsm.update(tab_detected)
        
        # Only trigger on initial detection with clear confidence
        if tab_updated is True and tab_detected:
            # Use a longer cooldown to prevent rapid tab switching
            # This makes the gesture more deliberate and prevents multiple tabs changing at once
            cooldown = config['gestures']['tab_switch'].get('dt_max', 0.55)  # Increase cooldown to slow down tab switching
            
            if state.can_trigger_gesture(cooldown):
                try:
                    # Add a small pause between key presses for more natural tab switching
                    if tab_direction:  # True = next tab (left swipe motion)
                        debug_print("Tab Forward - sending Command+Tab")
                        pyautogui.keyDown('command')
                        pyautogui.press('tab')
                        time.sleep(0.1)  # Brief pause for more controlled tab switching
                        pyautogui.keyUp('command')
                        state.update_gesture('Tab Forward')
                    else:  # False = previous tab (right swipe motion)
                        debug_print("Tab Back - sending Command+Shift+Tab")
                        pyautogui.keyDown('command')
                        pyautogui.keyDown('shift')
                        pyautogui.press('tab')
                        time.sleep(0.1)  # Brief pause for more controlled tab switching
                        pyautogui.keyUp('shift')
                        pyautogui.keyUp('command')
                        state.update_gesture('Tab Back')
                    
                    # Successfully changed tabs
                    switched_tab = True
                except Exception as e:
                    debug_print(f"Tab switch error: {e}")
    else:
        # Not in peace sign pose - immediately reset FSM
        state.tab_fsm.update(False)
    
    return switched_tab

def process_drawing_mode(hand_landmarks, frame):
    """
    Handle drawing mode based on victory gesture & draw with index finger
    
    Args:
        hand_landmarks: Hand landmarks from MediaPipe
        frame: Current video frame
        
    Returns:
        (toggled, updated_frame): Whether mode toggled and updated frame
    """
    toggled = False
    
    if hand_landmarks is None:
        state.victory_fsm.update(False)
        state.last_draw_point = None  # Reset drawing point
        return toggled, frame
    
    # Detect victory sign for mode toggle
    is_victory = detect_victory(hand_landmarks)
    victory_updated = state.victory_fsm.update(is_victory)
    
    if victory_updated is True:
        # Toggle between cursor and drawing mode
        new_mode = 'DRAWING' if state.mode == 'CURSOR' else 'CURSOR'
        state.update_mode(new_mode)
        toggled = True
    
    # Handle drawing if in drawing mode
    if state.mode == 'DRAWING':
        # Initialize drawing canvas if not already done
        if state.drawing_canvas is None or state.drawing_canvas.shape != frame.shape:
            state.drawing_canvas = np.zeros_like(frame)
        
        # Get index fingertip position for drawing
        index_tip = hand_landmarks.landmark[8]
        h, w = frame.shape[:2]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        
        # Draw only if index finger is extended
        if state.finger_states[1] == 1:
            # Draw line if previous point exists
            if state.last_draw_point is not None:
                cv2.line(
                    state.drawing_canvas,
                    state.last_draw_point,
                    (x, y),
                    config['drawing']['line_color'],
                    config['drawing']['line_thickness']
                )
            
            state.last_draw_point = (x, y)
        else:
            # Reset drawing point when index finger is not extended
            state.last_draw_point = None
        
        # Create a copy of the frame for drawing
        frame_copy = frame.copy()
        
        # Create mask and safely apply drawing overlay
        mask = cv2.cvtColor(state.drawing_canvas, cv2.COLOR_BGR2GRAY) > 0
        if np.any(mask):  # Only apply if mask has any True values
            try:
                # Use numpy's where to safely combine the images
                frame_copy = np.where(
                    mask[:, :, np.newaxis],
                    cv2.addWeighted(frame, 0.3, state.drawing_canvas, 0.7, 0),
                    frame_copy
                )
            except Exception as e:
                debug_print(f"Error in drawing overlay: {e}")
        
        return toggled, frame_copy
    
    return toggled, frame

def save_drawing():
    """Save current drawing to Desktop"""
    if state.mode != 'DRAWING' or state.drawing_canvas is None:
        print("[WARNING] Nothing to save. Drawing mode not active.")
        return False
    
    try:
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.expanduser(config['drawing']['save_path'])
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        filename = os.path.join(save_path, f"drawing_{timestamp}.png")
        
        # Save drawing
        cv2.imwrite(filename, state.drawing_canvas)
        print(f"[INFO] Drawing saved to: {filename}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save drawing: {e}")
        return False

def process_media_controls(hand_landmarks):
    """
    Process media control gestures (fist for play/pause)
    
    Args:
        hand_landmarks: Hand landmarks from MediaPipe
    
    Returns:
        controlled: Whether media was controlled
    """
    controlled = False
    
    if hand_landmarks is None:
        state.fist_fsm.update(False)
        return controlled
    
    # Detect fist for play/pause
    is_fist = detect_fist(hand_landmarks)
    fist_updated = state.fist_fsm.update(is_fist)
    
    if fist_updated is True:
        if state.can_trigger_gesture(0.8):  # Higher cooldown for media controls
            toggle_media_playback()
            state.update_gesture('Play/Pause')
            controlled = True
            debug_print("Media play/pause toggled")
    
    return controlled

def process_zoom(hand_landmarks, prev_hand_landmarks):
    """
    Process zoom in/out gesture with both hands in pinch position
    
    Args:
        hand_landmarks: List of current hand landmarks for both hands
        prev_hand_landmarks: List of previous hand landmarks for both hands
        
    Returns:
        zoomed: Whether zoom was performed
    """
    if (hand_landmarks[0] is None or hand_landmarks[1] is None or 
        prev_hand_landmarks[0] is None or prev_hand_landmarks[1] is None):
        return False
    
    # Check for zoom gesture (pinching with both hands)
    zoom_detected, zoom_in, magnitude = detect_zoom_gesture(
        hand_landmarks[0], 
        hand_landmarks[1],
        prev_hand_landmarks[0],
        prev_hand_landmarks[1],
        threshold=config['gestures'].get('zoom', {}).get('threshold', 0.04)
    )
    
    if zoom_detected:
        if state.can_trigger_gesture(0.1):  # Shorter cooldown for responsive zoom
            # Perform zoom action
            repeat_count = min(max(1, int(magnitude)), 3)  # Limit zoom magnitude
            
            for _ in range(repeat_count):
                if zoom_in:
                    pyautogui.hotkey(*config['mac_keys']['zoom_in'])
                    state.update_gesture('Zoom In')
                else:
                    pyautogui.hotkey(*config['mac_keys']['zoom_out'])
                    state.update_gesture('Zoom Out')
            
            debug_print(f"Zoom {'in' if zoom_in else 'out'}, magnitude: {magnitude}")
            return True
    
    return False

# --- Main Application Entry Point ---
def main():
    """Main function for AirAuth application"""
    # Setup camera
    cap = setup_camera()
    
    # Initialize face recognition data
    global face_data
    if not args.no_face_auth:
        face_data = load_face_data(config['face_recognition']['folder_path'])
        if not face_data:
            print("[WARNING] No face data found. You may want to enroll a user first.")
            print("          Use the --enroll flag: python app.py --enroll <username>")
            print("          Running with authentication disabled.")
            args.no_face_auth = True
    
    # Handle user enrollment if requested
    if args.enroll:
        print(f"[INFO] Starting enrollment for user: {args.enroll}")
        enroll_user(args.enroll, cap, config)
        if not args.no_face_auth:
            # Reload face data after enrollment
            face_data = load_face_data(config['face_recognition']['folder_path'])
            print(f"[INFO] Enrollment complete. Starting gesture control system...")

    # Main loop
    print("[INFO] Starting gesture control system...")
    print("[INFO] Press 'q' or 'Esc' to exit, 's' to save drawing in drawing mode")
    
    # For tracking both hands
    prev_hands_landmarks = [None, None]
    
    try:
        while not state.exiting:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                break
            
            # Flip frame horizontally for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process face recognition if authentication is enabled
            if not args.no_face_auth:
                user, face_distance = recognize_face(frame, face_data, config)
                locked, current_user, face_distance = lock_manager.update(user, face_distance)
                state.update_locked_state(locked, current_user, face_distance)
            
            # Process hand tracking if system is unlocked
            if args.no_face_auth or not state.locked:
                # Detect hand landmarks (both hands)
                hand_landmarks = process_hand_landmarks(frame)
                
                # Update state with primary hand landmarks
                state.hand_present = hand_landmarks[0] is not None
                state.hand_landmarks = hand_landmarks[0] if state.hand_present else None
                
                # Process two-handed gestures first (zoom)
                if hand_landmarks[0] is not None and hand_landmarks[1] is not None:
                    # We have both hands detected
                    if prev_hands_landmarks[0] is not None and prev_hands_landmarks[1] is not None:
                        # Try zoom gesture
                        zoomed = process_zoom(hand_landmarks, prev_hands_landmarks)
                        if zoomed:
                            # If zooming, skip other gestures
                            prev_hands_landmarks = hand_landmarks.copy()
                            
                            # Update FPS counter
                            state.update_fps(time.time())
                            display_frame = draw_ui(frame, state.ui_state, state.fps, debug=debug)
                            cv2.imshow('AirAuth: Gesture Control', display_frame)
                            
                            # Wait for key input
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == 27:  # 'q' or Esc
                                state.exiting = True
                            
                            continue  # Skip to next frame
                
                # Process single-hand gestures
                if state.hand_present:
                    # Update finger states for primary hand
                    state.update_finger_states(state.hand_landmarks)
                    
                    # Process gestures
                    cursor_pos = process_cursor_movement(state.hand_landmarks)
                    left_click, right_click = process_clicks(state.hand_landmarks)
                    
                    # Only process tab switching if hand was present in previous frame
                    if state.prev_hand is not None:
                        tab_switched = process_tab_switch(state.hand_landmarks, state.prev_hand)
                        # Media controls
                        media_controlled = process_media_controls(state.hand_landmarks)
                    
                    # Drawing mode toggle and drawing
                    drawing_toggled, frame = process_drawing_mode(state.hand_landmarks, frame)
                    
                else:
                    # Reset gesture state when no hand is detected
                    state.reset_gesture()
                
                # Update previous hand states
                state.prev_hand = state.hand_landmarks
                prev_hands_landmarks = hand_landmarks.copy()
            else:
                # Reset state when locked
                state.hand_present = False
                state.hand_landmarks = None
                state.prev_hand = None
                prev_hands_landmarks = [None, None]
                state.reset_gesture()
            
            # Update FPS counter
            state.update_fps(time.time())
            
            # Draw UI overlay
            display_frame = draw_ui(frame, state.ui_state, state.fps, debug=debug)
            
            # Show frame
            cv2.imshow('AirAuth: Gesture Control', display_frame)
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or Esc
                state.exiting = True
            elif key == ord('s') and state.mode == 'DRAWING':  # 's' in drawing mode
                saved = save_drawing()
                if saved:
                    state.ui_state['gesture'] = 'Drawing Saved'
            
            # Check for closed window
            if cv2.getWindowProperty('AirAuth: Gesture Control', cv2.WND_PROP_VISIBLE) < 1:
                state.exiting = True
            
            # Calculate remaining time for frame rate control
            elapsed = time.time() - loop_start
            target_frame_time = 1.0 / config['camera']['fps']
            remaining = target_frame_time - elapsed
            
            if remaining > 0:
                time.sleep(remaining)
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Clean up resources
        print("\n[INFO] Cleaning up resources...")
        try:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            print("[INFO] Resources released successfully")
        except Exception as e:
            print(f"[ERROR] Error during cleanup: {e}")

if __name__ == "__main__":
    main()