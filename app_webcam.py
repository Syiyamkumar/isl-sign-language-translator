import cv2
import json
import collections
import numpy as np
import torch
import mediapipe as mp
# Import WORDS_LABEL_MAP (or SENTENCES_LABEL_MAP if you trained sentences)
from config import SEQUENCE_LENGTH, DEVICE, MODEL_SAVE_PATH, WORDS_LABEL_MAP
from model import ISLBiLSTM
from extract_features import extract_keypoints

# Load labels and model
with open(WORDS_LABEL_MAP, "r") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

model = ISLBiLSTM(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
current_pred, conf = "Waiting...", 0.0

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        # Render landmarks for visual feedback
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        keypoints = extract_keypoints(results)
        sequence_buffer.append(keypoints)
        
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            input_tensor = torch.tensor(np.array([sequence_buffer]), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                best_idx = torch.argmax(probs, dim=1).item()
                conf = probs[0][best_idx].item()
                
                if conf > 0.70:
                    current_pred = idx_to_label[best_idx]
                    
        # UI Overlay
        cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)
        cv2.putText(frame, f"Sign: {current_pred} ({conf:.1%})", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("ISL Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()