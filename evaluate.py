# evaluate.py
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from config import DEVICE, MODEL_SAVE_PATH, WORDS_LABEL_MAP
from dataset import ISLDataset
from model import ISLBiLSTM

def evaluate():
    with open(WORDS_LABEL_MAP, "r") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}
    
    dataset = ISLDataset(mode="words")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = ISLBiLSTM(num_classes=len(label_map))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
            
    target_names = [idx_to_label[i] for i in range(len(label_map))]
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    evaluate()