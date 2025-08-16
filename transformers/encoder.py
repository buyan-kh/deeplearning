import torch
import torch.nn as nn
from attention import TransformerAttention
from ffn import FFN

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_head, dropout=0.1, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.att = TransformerAttention(
            d_model=d_model,
            num_head=num_head,
            dropout=dropout,
            bias=bias
        )
        self.ffn = FFN(
            d_model=d_model,
            d_ff=d_ff
        )
        self.dropout = nn.Dropout(p=dropout)
        self.LayerNorm_att = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)

    def forward(self, embed_input, padding_mask=None):
        batch_size, seq_len, _ = embed_input.size()
        att_sublayer = self.att(sequence=embed_input, key_value_states=None, att_mask=padding_mask)
        att_sublayer = self.dropout(att_sublayer)
        att_normalized = self.LayerNorm_att(embed_input + att_sublayer)
        ffn_sublayer = self.ffn(att_normalized)
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized + ffn_sublayer)
        return ffn_normalized
