import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_joints, frame_feature_dim=128, lstm_hidden_dim=256,
                 lstm_layers=2, bidirectional=False, fc_hidden_dim=128,
                 output_dim=402, dropout_rate=0.3):

        super(LSTM, self).__init__()
        self.num_joints = num_joints
        self.input_dim = num_joints * 3 
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.input_dim, frame_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(frame_feature_dim, frame_feature_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=frame_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_out_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        input [B, T, J, 3]
        output [B, T, 402]
        """
        B, T, J, D = x.size() 
        x = x.view(B, T, self.input_dim)

        x_enc = self.frame_encoder(x.view(B * T, self.input_dim))  
        x_enc = x_enc.view(B, T, -1)
       
        lstm_out, _ = self.lstm(x_enc) 

        out = self.fc(lstm_out.contiguous().view(B * T, self.lstm_out_dim)) 
        out = out.view(B, T, -1)
        return out
