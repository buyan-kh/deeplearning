import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_model // num_head
        self.dropout_rate = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(p=dropout)
        self.scaler = float(1.0 / math.sqrt(self.d_head))

    def forward(self, sequence, key_value_states=None, att_mask=None):
        batch_size, seq_len, model_dim = sequence.size()
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn\'t match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
                f"Cross attention key/value dimension {key_value_states.size(-1)} doesn\'t match model dimension {self.d_model}"

        is_cross_attention = key_value_states is not None
        Q_state = self.q_proj(sequence)
        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_len
            K_state = self.k_proj(sequence)
            V_state = self.v_proj(sequence)

        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head).transpose(1, 2)
        K_state = K_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1, 2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1, 2)

        Q_state = Q_state * self.scaler
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1, -2))

        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask

        att_score = F.softmax(self.att_matrix, dim=-1)
        att_score = self.dropout(att_score)
        att_output = torch.matmul(att_score, V_state)
        att_output = att_output.transpose(1, 2)
        att_output = att_output.contiguous().view(batch_size, seq_len, self.num_head * self.d_head)
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), \
            f"Final output shape {att_output.size()} incorrect"

        return att_output
