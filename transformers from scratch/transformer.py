import torch
import torch.nn as nn
from embedding import EmbeddingWithProjection
from encoder_decoder import TransformerEncoderDecoder

class Transformer(nn.Module):
    def __init__(self, num_layer, d_model, d_embed, d_ff, num_head, src_vocab_size, tgt_vocab_size, max_position_embeddings=512, dropout=0.1, bias=True):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        
        self.src_embedding = EmbeddingWithProjection(
            vocab_size=src_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        self.tgt_embedding = EmbeddingWithProjection(
            vocab_size=tgt_vocab_size,
            d_embed=d_embed,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )
        
        self.encoder_decoder = TransformerEncoderDecoder(
            num_layer=num_layer,
            d_model=d_model,
            d_ff=d_ff,
            num_head=num_head,
            dropout=dropout,
            bias=bias
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def shift_target_right(self, tgt_tokens):
        batch_size, seq_len = tgt_tokens.size()
        start_tokens = torch.zeros(batch_size, 1, dtype=tgt_tokens.dtype, device=tgt_tokens.device)
        shifted_tokens = torch.cat([start_tokens, tgt_tokens[:, :-1]], dim=1)
        return shifted_tokens
        
    def forward(self, src_tokens, tgt_tokens, padding_mask=None):
        shifted_tgt_tokens = self.shift_target_right(tgt_tokens)
        src_embedding = self.src_embedding(src_tokens)
        tgt_embedding = self.tgt_embedding(shifted_tgt_tokens)
        
        decoder_output = self.encoder_decoder(
            embed_encoder_input=src_embedding,
            embed_decoder_input=tgt_embedding,
            padding_mask=padding_mask
        )
        
        logits = self.output_projection(decoder_output)
        log_probs = self.softmax(logits)
        
        return log_probs
