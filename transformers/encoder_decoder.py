import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, num_layer, d_model, d_ff, num_head, dropout=0.1, bias=True):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias
        
        self.encoder_stack = nn.ModuleList([TransformerEncoder(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_head=self.num_head,
            dropout=self.dropout,
            bias=self.bias) for _ in range(self.num_layer)])

        self.decoder_stack = nn.ModuleList([TransformerDecoder(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_head=self.num_head,
            dropout=self.dropout,
            bias=self.bias) for _ in range(self.num_layer)])

    def forward(self, embed_encoder_input, embed_decoder_input, padding_mask=None):
        encoder_output = embed_encoder_input
        for encoder in self.encoder_stack:
            encoder_output = encoder(encoder_output, padding_mask)
        
        decoder_output = embed_decoder_input
        for decoder in self.decoder_stack:
            decoder_output = decoder(decoder_output, encoder_output, padding_mask)
        
        return decoder_output
