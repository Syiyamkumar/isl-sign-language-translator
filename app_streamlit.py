# app_streamlit.py
import streamlit as st
import cv2
import json
import numpy as np
import torch
import mediapipe as mp
import collections
from config import DEVICE, MODEL_SAVE_PATH, WORDS_LABEL_MAP, SEQUENCE_LENGTH
from model import ISLBiLSTM
from extract_features import extract_keypoints

st.set_page_config(page_title="ISL Sign Language Recognition", layout="wide")
st.title("🤟 Indian Sign Language (ISL) Recognition System")

@st.cache_resource
def load_system():
    with open(WORDS_LABEL_MAP, "r") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    
    model = ISLBiLSTM(num_classes=len(label_map))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, idx_to_label

model, idx_to_label = load_system()
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

mode = st.sidebar.selectbox("Choose Mode", ["Live Webcam", "Upload Video File"])

if mode == "Live Webcam":
    start = st.button("Start Camera")
    stop = st.button("Stop Camera")
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    if start:
        cap = cv2.VideoCapture(0)
        sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
                sequence.append(extract_keypoints(results))
                
                if len(sequence) == SEQUENCE_LENGTH:
                    tensor_in = torch.tensor(np.array([sequence]), dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        probs = torch.softmax(model(tensor_in), dim=1)
                        best_idx = torch.argmax(probs, dim=1).item()
                        conf = probs[0][best_idx].item()
                        
                        if conf > 0.70:
                            status_text.markdown(f"### Detected: **{idx_to_label[best_idx]}** ({conf:.1%})")
                
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

elif mode == "Upload Video File":
    uploaded = st.file_uploader("Upload an ISL Gesture Video (.mp4)", type=["mp4", "avi", "mov"])
    if uploaded:
        temp_path = "temp_input.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
            
        cap = cv2.VideoCapture(temp_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        st.video(temp_path)
        if len(frames) > 0:
            indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
            seq = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for idx in indices:
                    rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)
                    seq.append(extract_keypoints(results))
                    
            tensor_in = torch.tensor(np.array([seq]), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(tensor_in), dim=1)
                best_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0][best_idx].item()
                
            st.success(f"Predicted Sign: **{idx_to_label[best_idx]}** | Confidence: **{conf:.2%}**")