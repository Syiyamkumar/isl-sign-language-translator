import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from config import (
    WORDS_FRAMES_DIR,
    SENTENCES_FRAMES_DIR,
    SENTENCES_VIDEOS_DIR,
    PROCESSED_WORDS,
    PROCESSED_SENTENCES,
    WORDS_LABEL_MAP,
    SENTENCES_LABEL_MAP,
    SEQUENCE_LENGTH
)

VALID_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VALID_VID_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])

def process_frame_list(img_paths, holistic):
    if len(img_paths) == 0:
        return None
    indices = np.linspace(0, len(img_paths) - 1, SEQUENCE_LENGTH, dtype=int)
    sequence = []
    for idx in indices:
        img = cv2.imread(img_paths[idx])
        if img is None:
            sequence.append(np.zeros(225))
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        sequence.append(extract_keypoints(results))
    return np.array(sequence, dtype=np.float32)

def process_video_file(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        return None
        
    indices = np.linspace(0, len(frames) - 1, SEQUENCE_LENGTH, dtype=int)
    sequence = []
    for idx in indices:
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        sequence.append(extract_keypoints(results))
    return np.array(sequence, dtype=np.float32)

def extract_directory(source_dir, output_dir, label_map_file, is_video=False):
    if not os.path.exists(source_dir):
        print(f"[!] Directory not found: {source_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])
    
    label_map = {name: idx for idx, name in enumerate(classes)}
    with open(label_map_file, "w") as f:
        json.dump(label_map, f, indent=4)

    print(f"\nProcessing {len(classes)} classes from: {os.path.basename(source_dir)}")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for class_name in tqdm(classes, desc=f"Extracting {os.path.basename(source_dir)}"):
            class_in_path = os.path.join(source_dir, class_name)
            class_out_path = os.path.join(output_dir, class_name)
            os.makedirs(class_out_path, exist_ok=True)
            
            sample_idx = 0
            
            if is_video:
                # Video files extraction (.mp4)
                for root, _, files in os.walk(class_in_path):
                    for file in files:
                        if file.lower().endswith(VALID_VID_EXTS):
                            vid_path = os.path.join(root, file)
                            seq = process_video_file(vid_path, holistic)
                            if seq is not None:
                                np.save(os.path.join(class_out_path, f"vid_sample_{sample_idx}.npy"), seq)
                                sample_idx += 1
            else:
                # Image frames extraction
                for root, dirs, files in os.walk(class_in_path):
                    img_files = sorted([os.path.join(root, f) for f in files if f.lower().endswith(VALID_IMG_EXTS)])
                    if len(img_files) > 0:
                        seq = process_frame_list(img_files, holistic)
                        if seq is not None:
                            np.save(os.path.join(class_out_path, f"frame_sample_{sample_idx}.npy"), seq)
                            sample_idx += 1

if __name__ == "__main__":
    # 1. Extract Words (Frames_Word_Level)
    extract_directory(WORDS_FRAMES_DIR, PROCESSED_WORDS, WORDS_LABEL_MAP, is_video=False)
    
    # 2. Extract Sentences from Frames (Frames_Sentence_Level)
    extract_directory(SENTENCES_FRAMES_DIR, PROCESSED_SENTENCES, SENTENCES_LABEL_MAP, is_video=False)
    
    # 3. Extract Sentences from Videos (Videos_Sentence_Level)
    extract_directory(SENTENCES_VIDEOS_DIR, PROCESSED_SENTENCES, SENTENCES_LABEL_MAP, is_video=True)
    
    print("\nExtraction complete! All frame sequences and video gestures converted to .npy coordinates.")