import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, MODEL_SAVE_PATH
from dataset import ISLDataset
from model import ISLBiLSTM

# Select target mode: "words" (131 classes) or "sentences" (101 classes)
TRAIN_MODE = "words"

def train():
    dataset = ISLDataset(mode=TRAIN_MODE)
    total_samples = len(dataset)
    print(f"Total dataset sequences loaded: {total_samples}")
    
    if total_samples == 0:
        print("[!] No extracted samples found. Run extract_features.py first.")
        return

    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    num_classes = len(dataset.label_map)
    model = ISLBiLSTM(num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            
        train_acc = train_correct / len(train_ds) if len(train_ds) > 0 else 0
        
        # Validation Loop
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_ds) if len(val_ds) > 0 else 0
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Loss: {train_loss:.4f} - Train Acc: {train_acc:.1%} - Val Acc: {val_acc:.1%}")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining Complete! Checkpoint saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()