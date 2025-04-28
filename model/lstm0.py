import torch
import torch.nn as nn

class LSTM0(nn.Module):
    def __init__(self):
        super(LSTM0, self).__init__()
        self.input_dim = 60      
        self.hidden_dim = 256    
        self.output_dim = 402   

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        batch, seq, j, d = x.size()
        assert j * d == self.input_dim, f"Expected flattened size of {self.input_dim}, got {j*d}"

        x = x.view(batch, seq, -1)

        lstm_out, _ = self.lstm(x)

        output = self.fc(lstm_out.contiguous().view(-1, self.hidden_dim))

        output = output.view(batch, seq, self.output_dim)
        return output