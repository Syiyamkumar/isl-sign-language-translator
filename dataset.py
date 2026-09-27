import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from config import PROCESSED_WORDS, PROCESSED_SENTENCES, WORDS_LABEL_MAP, SENTENCES_LABEL_MAP

class ISLDataset(Dataset):
    def __init__(self, mode="words"):
        """
        mode: 'words' or 'sentences'
        """
        if mode == "words":
            data_dir = PROCESSED_WORDS
            map_path = WORDS_LABEL_MAP
        else:
            data_dir = PROCESSED_SENTENCES
            map_path = SENTENCES_LABEL_MAP
            
        with open(map_path, "r") as f:
            self.label_map = json.load(f)
            
        self.samples = []
        for class_name, label in self.label_map.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for npy_file in os.listdir(class_dir):
                if npy_file.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, npy_file), label))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)