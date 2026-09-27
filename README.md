# Real-Time Indian Sign Language (ISL) Recognition & Translation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-007ACC?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-In%20Active%20Development-success.svg)]()

An end-to-end Computer Vision and Deep Learning system designed to bridge communication barriers by recognizing and translating Indian Sign Language (ISL) gestures in real time.

---

## 📌 Project Overview
- **Keypoint Extraction:** Extracts 225-dimensional spatial-temporal keypoints (Pose, Left Hand, Right Hand) per frame using Google MediaPipe Holistic.
- **Sequence Modeling:** Temporal modeling with a 2-layer Bidirectional LSTM (BiLSTM) network over 30-frame gesture sequences.
- **Inference Interfaces:** Features an OpenCV real-time desktop overlay and an interactive Streamlit web dashboard supporting live video streams and video uploads.
- **Ongoing Roadmap:** Integrating Generative AI to map recognized sequence tokens into coherent natural-language sentences translated across multiple Indian regional languages.

---

## 🏗️ Architecture & Pipeline

1. **Feature Extraction (`extract_features.py`):** 
   - Normalizes video sequences to 30 frames.
   - Extracts coordinates: Pose (33 × 3) + Left Hand (21 × 3) + Right Hand (21 × 3) = 225 features per frame.
2. **Model Training (`train.py` & `model.py`):**
   - Architecture: 2-Layer BiLSTM (`hidden_dim=128`, `dropout=0.3`) followed by BatchNorm, ReLU, and linear projection.
   - Optimized with AdamW and Cross-Entropy Loss.
3. **Inference (`app_streamlit.py` / `app_webcam.py`):**
   - Continuously maintains a 30-frame rolling sequence buffer via a `deque`.
   - Fires softmax predictions when confidence exceeds the 70% threshold.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Syiyamkumar/isl-sign-language-translator.git](https://github.com/Syiyamkumar/isl-sign-language-translator.git)
   cd isl-sign-language-translator
