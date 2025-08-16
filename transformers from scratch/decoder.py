import torch
import torch.nn as nn
from attention import TransformerAttention
from ffn import FFN

class TransformerDecoder(nn.Module):
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
        self.LayerNorm_att1 = nn.LayerNorm(self.d_model)
        self.LayerNorm_att2 = nn.LayerNorm(self.d_model)
        self.LayerNorm_ffn = nn.LayerNorm(self.d_model)

    @staticmethod
    def create_causal_mask(seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, embed_input, cross_input, padding_mask=None):
        batch_size, seq_len, _ = embed_input.size()
        assert embed_input.size(-1) == self.d_model, f"Input dimension {embed_input.size(-1)} doesn't match model dimension {self.d_model}"
        assert cross_input.size(-1) == self.d_model, "Encoder output dimension doesn't match model dimension"

        causal_mask = self.create_causal_mask(seq_len).to(embed_input.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        att_sublayer1 = self.att(sequence=embed_input, key_value_states=None, att_mask=causal_mask)
        att_sublayer1 = self.dropout(att_sublayer1)
        att_normalized1 = self.LayerNorm_att1(embed_input + att_sublayer1)

        att_sublayer2 = self.att(sequence=att_normalized1, key_value_states=cross_input, att_mask=padding_mask)
        att_sublayer2 = self.dropout(att_sublayer2)
        att_normalized2 = self.LayerNorm_att2(att_normalized1 + att_sublayer2)

        ffn_sublayer = self.ffn(att_normalized2)
        ffn_sublayer = self.dropout(ffn_sublayer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized2 + ffn_sublayer)
        return ffn_normalized