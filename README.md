# Real-Time Indian Sign Language (ISL) Recognition and Translation System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-007ACC?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-blue.svg)]()

An end-to-end computer vision and deep learning framework engineered to facilitate accessible communication by performing real-time spatial feature extraction, temporal sequence classification, and multi-platform inference for Indian Sign Language (ISL).

---

## System Overview

* **Spatio-Temporal Feature Representation:** Extracts a 225-dimensional coordinate vector per frame using Google MediaPipe Holistic, capturing upper-body skeletal topology alongside bilateral hand articulations.
* **Recurrent Sequence Modeling:** Employs a 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network with dropout regularization to capture forward and backward temporal dynamics over 30-frame temporal windows.
* **Dual Inference Framework:** Deploys a lightweight OpenCV engine for low-latency desktop execution alongside an interactive Streamlit web dashboard supporting live video streams and batch file analysis.
* **Extensible Language Generation:** Designed for downstream integration with Large Language Models (LLMs) to synthesize recognized token sequences into grammatically coherent sentences across multiple regional languages.

---

## System Architecture

```text
[Input Video / Camera Feed]
            │
            ▼
[MediaPipe Holistic Pipeline]
 ├── Pose Landmarks:       33 × 3 (x, y, z)
 ├── Left Hand Landmarks:  21 × 3 (x, y, z)
 └── Right Hand Landmarks: 21 × 3 (x, y, z)
            │ (225-Dimensional Spatial Vector)
            ▼
[Rolling Temporal Buffer] ──> Uniform 30-Frame Sequence (FIFO Deque)
            │
            ▼
[BiLSTM Classification Engine]
 ├── 2-Layer Bidirectional LSTM (Hidden Dim: 128, Dropout: 0.3)
 ├── Batch Normalization & ReLU Non-Linearity
 └── Linear Classification Head
            │
            ▼
[Softmax Posterior Probabilities] ──> Confidence Threshold (> 0.70)
            │
            ▼
[Application Layer] ──> OpenCV Real-Time HUD / Streamlit Dashboard

---

## 🗺️ Project Milestones
- [x] Multi-landmark spatial extraction pipeline via MediaPipe Holistic
- [x] PyTorch BiLSTM gesture sequence classifier
- [x] Real-time sliding window inference engine
- [x] Multi-mode Streamlit dashboard (Webcam & Video upload)
- [ ] Sentence-level continuous sign translation
- [ ] Generative AI multilingual text translation
