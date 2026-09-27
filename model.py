import torch
import torch.nn as nn

class ISLBiLSTM(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=128, num_classes=131, num_layers=2):
        super(ISLBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output of the final time step
        final_state = lstm_out[:, -1, :]
        return self.fc(final_state)