# 🤟 Real-Time Indian Sign Language (ISL) Recognition & Translation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-007ACC?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-In%20Active%20Development-success.svg)]()

An end-to-end Computer Vision and Deep Learning system designed to bridge communication barriers by recognizing and translating Indian Sign Language (ISL) gestures in real time.

---

## 📌 Project Overview
- **Keypoint Extraction:** Extracts 225-dimensional spatio-temporal keypoints (Pose, Left Hand, Right Hand) per frame using Google MediaPipe Holistic.
- **Sequence Modeling:** Temporal sequence modeling using a 2-layer Bidirectional LSTM (BiLSTM) over 30-frame gesture windows.
- **Inference Interfaces:** Features an OpenCV real-time desktop overlay and an interactive Streamlit web application supporting live webcam streams and pre-recorded video uploads.
- **Ongoing Roadmap:** Integrating Generative AI to map recognized sequence tokens into coherent natural-language sentences translated across multiple Indian regional languages.

---

## 🏗️ Architecture & Pipeline

```text
[Input Video / Camera]
        │
        ▼
[MediaPipe Holistic] ──> Pose (33×3) + LH (21×3) + RH (21×3) = 225 Keypoints
        │
        ▼
[Sequence Buffer]   ──> Uniform 30-Frame Rolling Window (FIFO Deque)
        │
        ▼
[BiLSTM Network]    ──> 2 Layers (Hidden Dim: 128, Dropout: 0.3)
        │
        ▼
[Softmax Output]    ──> Predicted Sign Token (>70% Confidence Threshold)
        │
        ▼
[UI / Overlay]      ──> Real-Time Label Display (OpenCV / Streamlit)
