import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.fc2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, input):
        batch_size, seq_length, d_input = input.size()
        assert self.d_model == d_input, "d_model must be the same dimension as the input"
        f1 = F.relu(self.fc1(input))
        f2 = self.fc2(f1)
        return f2
