# AirAuth 🖐️✨

> **Control your computer with gestures, secured by your face.**

## What is AirAuth?

AirAuth is a touchless computer control system that combines the magic of hand gestures with the security of facial recognition. Wave goodbye to your mouse and keyboard as you navigate your Mac using intuitive hand movements – but only after your face has been authenticated.

<p align="center">
  <img src="https://github.com/VanshikWaghela/AirAuth/assets/yourAssetID/airauth_demo.gif" alt="AirAuth Demo" width="600">
</p>

## ✨ Key Features

- **Touchless Control**: Navigate your Mac with natural hand gestures
- **Face Authentication**: Biometric security layer ensures only authorized users can control the system
- **Drawing Mode**: Switch to a virtual canvas with a simple gesture
- **Ultra-Smooth Cursor**: Advanced filtering algorithms provide precise cursor control
- **Rich Gesture Set**: From clicks to scrolls, tab switching to media controls
- **Multi-user Support**: Enroll multiple users for shared access

## 🖐️ Gesture Library

| Gesture | Action |
|---------|--------|
| 👆 Index finger | Move cursor |
| 👌 Thumb-index pinch | Left click |
| 👌 Thumb-middle pinch | Right click |
| ✌️ Victory sign | Toggle drawing mode |
| 🖐️ Open palm up/down | Scroll |
| ✌️ Two-finger swipe | Switch tabs |
| 👊 Closed fist | Play/pause media |
| 🤏🤏 Two-hand pinch | Zoom in/out |

## 🚀 Quick Start

### Prerequisites
- macOS
- Webcam
- Python with Conda environment

### Installation

```bash
# Clone the repository
git clone https://github.com/VanshikWaghela/AirAuth.git
cd AirAuth

# Create and activate conda environment
conda create -n airauth python=3.9
conda activate airauth

# Install dependencies
pip install -r requirements.txt
```

### First-time Setup

```bash
# Enroll your face (required before first use)
python app.py --enroll your_name
```

### Launch AirAuth

```bash
# Start with standard settings
python app.py

# Or enable debug mode
python app.py --debug
```

## 🔒 Security First

AirAuth uses computer vision to create a security layer that only enables gesture control for recognized users:

- **Facial Recognition**: Uses face_recognition library with HOG or CNN models
- **Adaptive Security**: Configurable tolerance levels for different security needs
- **Grace Period**: Brief window of continued access if your face is temporarily not visible

## 💡 Tips for Best Results

- Position yourself 50-70cm from the webcam
- Ensure consistent, even lighting
- Keep your hand movements deliberate and within the frame
- Start with small gestures until you get comfortable

## 🛠️ Configuration

Fine-tune AirAuth by editing `config.yaml`:

```yaml
# Example configurations
detection:
  min_confidence: 0.75

gestures:
  left_click:
    hold_frames: 1
    threshold: 0.06
    
face_recognition:
  model: "hog"  # "hog" (faster) or "cnn" (more accurate)
  tolerance: 0.5
```

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests


<p align="center">Made with ❤️ by <a href="https://github.com/VanshikWaghela">Vanshik Waghela</a></p>


