#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# --- macOS Permissions Check ---
if platform.system() == 'Darwin':
    # Accessibility check (mouse/keyboard control)
    try:
        import Quartz
        if not Quartz.CGPreflightListenEventAccess():
            print("[ERROR] Accessibility permissions not granted.\nGo to System Settings > Privacy & Security > Accessibility, and add Terminal or your Python interpreter.")
            sys.exit(1)
    except ImportError:
        print("[WARNING] Quartz not installed; skipping Accessibility check.")
    # Camera check
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera access denied or unavailable.\nGo to System Settings > Privacy & Security > Camera, and allow access for Terminal or your Python interpreter.")
        sys.exit(1)
    cap.release()

# --- Config and Argument Parsing ---
def load_config_safe(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Could not load config: {e}. Using defaults.")
        return load_config('config.yaml')

def load_calib_safe(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

# --- Debug utility ---
def debug_print(msg):
    if debug:
        print(f"[DEBUG] {msg}")

parser = argparse.ArgumentParser(description='Gesture Control System')
parser.add_argument('--config', default='config.yaml', help='Path to config file')
parser.add_argument('--calib', default='calib.json', help='Path to calibration file')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--camera', type=int, default=0, help='Camera index')
parser.add_argument('--enroll', type=str, help='Enroll a new user with the specified username')
args = parser.parse_args()

config = load_config_safe(args.config)
calib = load_calib_safe(args.calib)

def get_smoothing_filters(config):
    method = config['smoothing']['method']
    params = config['smoothing']['params']
    if method == 'one_euro':
        from utils import OneEuroFilter
        return (OneEuroFilter(mincutoff=params['mincutoff'], beta=params['beta']),
                OneEuroFilter(mincutoff=params['mincutoff'], beta=params['beta']))
    else:
        from utils import KalmanFilter
        return (KalmanFilter(), KalmanFilter())

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=config['detection']['min_confidence'],
    min_tracking_confidence=config['detection'].get('min_confidence', 0.7),
    model_complexity=0,  # Lowest complexity for speed
    static_image_mode=False  # Tracking mode for better performance
)

# --- Screen Setup ---
screen_width, screen_height = pyautogui.size()

# --- Smoothing Filter ---
cursor_filter_x, cursor_filter_y = get_smoothing_filters(config)

# --- State ---
mode = 'CURSOR'  # Modes: CURSOR, DRAWING, SCROLL, VOLUME
ui_state = {'mode': 'Cursor', 'gesture': None, 'show_fps': True, 'error': None}
drawing_canvas = None
prev_mouse_pos = (screen_width//2, screen_height//2)
prev_hand_landmarks = [None, None]
click_cooldown = 0
last_gesture_time = 0
fps = 0
prev_time = time.time()
debug = args.debug

# --- Modular gesture detection ---
def detect_gestures(hand_landmarks, prev_hand_landmarks, config, now, last_gesture_time, mode):
    """
    Returns: (gesture, action_dict, new_mode, debug_msgs)
    """
    gesture = None
    action = {}
    new_mode = mode
    debug_msgs = []
    if hand_landmarks[0] is not None:
        fs = get_finger_states(hand_landmarks[0])
        debug_msgs.append(f"Finger states: {fs}")
        # Neutral
        if fs == [1,1,1,1,1]:
            gesture = 'neutral'
            debug_msgs.append("Detected: Neutral/Reset")
        # Cursor
        elif fs[1] == 1 and sum(fs[2:]) == 0:
            gesture = 'cursor'
            debug_msgs.append("Detected: Cursor Move")
        # Tab Switch (V sign)
        elif fs[1] == 1 and fs[2] == 1 and sum(fs) == 2:
            gesture = 'tab_switch'
            debug_msgs.append("Detected: Tab Switch (V sign)")
        # Scroll
        elif sum(fs) >= 4:
            gesture = 'scroll'
            debug_msgs.append("Detected: Scroll (Open Palm)")
        # Drawing Mode (V sign)
        if detect_victory(hand_landmarks[0], min_dist=0.08, min_angle=18):
            gesture = 'drawmode_toggle'
            debug_msgs.append("Detected: Drawing Mode Toggle (Victory)")
        # Fist
        if get_finger_states(hand_landmarks[0]) == [0,0,0,0,0]:
            gesture = 'fist'
            debug_msgs.append("Detected: Fist (Play/Pause)")
    # Two-handed zoom
    if hand_landmarks[0] is not None and hand_landmarks[1] is not None and prev_hand_landmarks[0] is not None and prev_hand_landmarks[1] is not None:
        is_zoom, zoom_in = detect_zoom(hand_landmarks[0], hand_landmarks[1], prev_hand_landmarks[0], prev_hand_landmarks[1])
        if is_zoom:
            gesture = 'zoom'
            action['zoom_in'] = zoom_in
            debug_msgs.append(f"Detected: Zoom {'In' if zoom_in else 'Out'}")
    return gesture, action, new_mode, debug_msgs

# --- Face Recognition Setup ---
face_data = load_face_data(config['face_recognition']['folder_path'])
lock_manager = LockStateManager(config)

# --- Main Loop ---
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit(1)

# Set camera properties for better quality while maintaining performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution for better quality
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Higher resolution for better quality
cap.set(cv2.CAP_PROP_FPS, 30)  # Request higher FPS if camera supports it
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Slightly increase brightness
cap.set(cv2.CAP_PROP_CONTRAST, 150)  # Slightly increase contrast

# Handle user enrollment if requested
if args.enroll:
    enroll_user(args.enroll, cap, config)
    # Reload face data after enrollment
    face_data = load_face_data(config['face_recognition']['folder_path'])
    print(f"[INFO] Enrollment complete. Starting gesture control system...")

try:
    # --- Gesture FSMs for debounce/reliability ---
    pinch_fsm = GestureFSM(hold_frames=config['gestures']['left_click']['hold_frames'])
    rpinch_fsm = GestureFSM(hold_frames=config['gestures']['right_click']['hold_frames'])
    scroll_fsm = GestureFSM(hold_frames=3)
    tab_fsm = GestureFSM(hold_frames=config['gestures']['tab_switch']['hold_frames'])
    fist_fsm = GestureFSM(hold_frames=3)
    neutral_fsm = GestureFSM(hold_frames=3)
    victory_fsm = GestureFSM(hold_frames=5)
    last_draw = None
    cursor_history_x = deque(maxlen=3)  # Lowered for more responsive smoothing
    cursor_history_y = deque(maxlen=3)
    last_cursor = (screen_width//2, screen_height//2)
    
    # Frame skipping for hand detection
    frame_count = 0
    hand_detection_interval = 1  # Process every 2nd frame
    last_valid_results = None
    
    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            ui_state['error'] = 'Camera read error.'
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        # Face recognition processing (already optimized with threading in utils.py)
        user, face_distance = recognize_face(frame, face_data, config)
        locked, current_user, face_distance = lock_manager.update(user, face_distance)
        
        # Update UI state with lock information
        ui_state['locked'] = locked
        ui_state['user'] = current_user
        ui_state['face_distance'] = face_distance
        
        # Process hand landmarks only if system is unlocked
        # Use frame skipping for hand detection to improve performance
        frame_count = (frame_count + 1) % (hand_detection_interval + 1)
        
        if not locked and frame_count == 0:
            # Process at reduced resolution for better performance
            small_rgb = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            small_rgb = cv2.cvtColor(small_rgb, cv2.COLOR_BGR2RGB)
            results = hands.process(small_rgb)
            if results.multi_hand_landmarks:
                last_valid_results = results
        elif not locked and last_valid_results and last_valid_results.multi_hand_landmarks:
            # Use the last valid results for skipped frames
            results = last_valid_results
        else:
            results = None
            
        hand_landmarks = [None, None]
        if results and results.multi_hand_landmarks:
            for i, lm in enumerate(results.multi_hand_landmarks[:2]):
                hand_landmarks[i] = lm
                # Only draw landmarks if explicitly enabled and not locked
                if config['ui']['show_hand_landmarks'] and not locked and debug:
                    # Draw with simplified style for better performance
                    mp_drawing.draw_landmarks(
                        frame, 
                        lm, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                    )
        now = time.time()
        gesture_cooldown = config['gestures'].get('cooldown_s', 0.2)  # Lowered for snappier response
        gesture, action, new_mode, debug_msgs = detect_gestures(hand_landmarks, prev_hand_landmarks, config, now, last_gesture_time, mode)
        for msg in debug_msgs:
            debug_print(msg)
        # --- Main gesture logic ---
        if hand_landmarks[0] is not None and not locked:
            fs = get_finger_states(hand_landmarks[0])
            # --- Cursor (Index Extended) ---
            if fs == [0,1,0,0,0]:
                # Cursor mode: index only
                gesture = 'cursor'
                ui_state['gesture'] = 'Cursor Move'
                # Enhanced cursor movement with improved adaptive smoothing
                index_tip = hand_landmarks[0].landmark[8]
                
                # Get raw cursor position
                raw_x = index_tip.x * screen_width
                raw_y = index_tip.y * screen_height
                
                # Apply enhanced smoothing filter
                x = int(cursor_filter_x.apply(raw_x))
                y = int(cursor_filter_y.apply(raw_y))
                
                # Calculate movement speed for adaptive smoothing
                if cursor_history_x and cursor_history_y:
                    dx = abs(x - cursor_history_x[-1])
                    dy = abs(y - cursor_history_y[-1])
                    movement_speed = dx + dy
                    
                    # Enhanced adaptive duration based on movement speed
                    # Completely eliminate duration for small movements for better responsiveness
                    # Use minimal duration for larger movements to reduce jitter
                    if movement_speed < 15:
                        duration = 0  # Instant movement for small adjustments
                    elif movement_speed < 40:
                        duration = 0.002  # Very slight smoothing for medium movements
                    else:
                        duration = 0.004  # Minimal smoothing for large movements
                else:
                    duration = 0
                
                # Apply dead zone to prevent tiny jitters when trying to hold still
                dead_zone = config['gestures']['cursor'].get('dead_zone', 0.04) * screen_width / 20
                if cursor_history_x and cursor_history_y:
                    if movement_speed < dead_zone:
                        # Use previous position if movement is within dead zone
                        x, y = cursor_history_x[-1], cursor_history_y[-1]
                
                # Move cursor with optimized parameters
                pyautogui.moveTo(x, y, duration=duration)
                
                # Update history
                last_cursor = (x, y)
                cursor_history_x.append(x)
                cursor_history_y.append(y)
                prev_mouse_pos = (x, y)
            # --- Left Click (Pinch: index only, robust) ---
            if detect_pinch(hand_landmarks[0], which='index'):
                if pinch_fsm.update(True) and now - last_gesture_time > config['gestures']['left_click']['debounce_s']:
                    pyautogui.click(button='left')
                    last_gesture_time = now
                    ui_state['gesture'] = 'Left Click'
            else:
                pinch_fsm.update(False)
            # --- Right Click (Pinch: index+middle, robust) ---
            if detect_pinch(hand_landmarks[0], which='middle'):
                if rpinch_fsm.update(True) and now - last_gesture_time > config['gestures']['right_click']['debounce_s']:
                    pyautogui.click(button='right')
                    last_gesture_time = now
                    ui_state['gesture'] = 'Right Click'
            else:
                rpinch_fsm.update(False)
            # --- Scroll (L-shape: thumb+index extended, vertical motion) ---
            if prev_hand_landmarks[0] is not None:
                scroll_detected, scroll_dir = detect_scroll(hand_landmarks[0], prev_hand_landmarks[0])
                if scroll_detected:
                    if scroll_fsm.update(True) and now - last_gesture_time > config['gestures']['scroll']['dt_max']:
                        pyautogui.scroll(int(scroll_dir * config['gestures']['scroll']['dy_thresh']))
                        last_gesture_time = now
                        ui_state['gesture'] = 'Scroll'
                else:
                    scroll_fsm.update(False)
            # --- Tab Switch (Two-finger horizontal swipe: index+middle extended, horizontal swipe) ---
            if prev_hand_landmarks[0] is not None:
                tab_detected, tab_dir = detect_tab_switch(hand_landmarks[0], prev_hand_landmarks[0])
                if tab_detected:
                    if tab_fsm.update(True) and now - last_gesture_time > config['gestures']['tab_switch']['dt_max']:
                        if tab_dir:
                            pyautogui.hotkey(*config['mac_keys']['tab_forward'])
                        else:
                            pyautogui.hotkey(*config['mac_keys']['tab_back'])
                        last_gesture_time = now
                        ui_state['gesture'] = 'Tab Switch'
                else:
                    tab_fsm.update(False)
            # --- Neutral (Open palm) ---
            if detect_neutral(hand_landmarks[0]):
                if neutral_fsm.update(True):
                    ui_state['gesture'] = 'Neutral'
            else:
                neutral_fsm.update(False)
            # --- Drawing Mode (Victory sign: index+middle extended, separated, robust) ---
            if victory_fsm.update(detect_victory(hand_landmarks[0], min_dist=0.08, min_angle=18)) and (now - last_gesture_time > 0.7):
                mode = 'DRAWING' if mode != 'DRAWING' else 'CURSOR'
                ui_state['mode'] = mode.capitalize()
                ui_state['gesture'] = 'Drawing Mode Toggled'
                last_gesture_time = now
                gesture = 'drawmode'
                debug_print(f"Drawing mode toggled: {mode}")
            # --- Drawing (Index Extended in Drawing Mode) ---
            if mode == 'DRAWING' and fs[1] == 1 and sum(fs[2:]) == 0:
                if drawing_canvas is None or drawing_canvas.shape != frame.shape:
                    drawing_canvas = np.zeros_like(frame)
                x = int(hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                y = int(hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                if last_draw is not None:
                    cv2.line(drawing_canvas, last_draw, (x, y), (0,0,255), 3)
                last_draw = (x, y)
                mask = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY) > 0
                frame[mask] = cv2.addWeighted(frame, 0.5, drawing_canvas, 0.5, 0)[mask]
            else:
                last_draw = None
        # --- Two-Handed Gestures (Zoom) ---
        if hand_landmarks[0] is not None and hand_landmarks[1] is not None and prev_hand_landmarks[0] is not None and prev_hand_landmarks[1] is not None:
            is_zoom, zoom_in = detect_zoom(hand_landmarks[0], hand_landmarks[1], prev_hand_landmarks[0], prev_hand_landmarks[1])
            if is_zoom and (now - last_gesture_time > gesture_cooldown):
                key_combo = config['mac_keys']['zoom_in'] if zoom_in else config['mac_keys']['zoom_out']
                pyautogui.hotkey(*key_combo)
                ui_state['gesture'] = 'Zoom In' if zoom_in else 'Zoom Out'
                last_gesture_time = now
                gesture = 'zoom'
                debug_print(f"Zoom triggered: {'In' if zoom_in else 'Out'}")
        # --- Fist (Play/Pause) ---
        if hand_landmarks[0] is not None and fist_fsm.update(detect_fist(hand_landmarks[0])) and (now - last_gesture_time > gesture_cooldown):
            toggle_media_playback()
            ui_state['gesture'] = 'Play/Pause'
            last_gesture_time = now
            gesture = 'fist'
            debug_print("Play/Pause triggered")
        # --- Error Handling ---
        if hand_landmarks[0] is None:
            ui_state['error'] = 'No hand detected.'
        else:
            ui_state['error'] = None
        prev_hand_landmarks = hand_landmarks.copy();
        # --- FPS & UI Overlay ---
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1/(now - prev_time)) if fps else 1/(now - prev_time)
        prev_time = now
        ui_state['fps'] = fps
        
        # Only draw UI at reduced resolution to save processing time
        display_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8) if not debug else frame
        display_frame = draw_ui(display_frame, ui_state, fps, debug=debug)
        cv2.imshow('Gesture Control System', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()