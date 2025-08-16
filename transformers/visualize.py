import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from decoder import TransformerDecoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def visualize_positional_encoding(seq_length=30, d_model=32):
    pe = np.zeros((seq_length, d_model))
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    plt.figure(figsize=(15, 8))
    for dim in range(8):
        plt.plot(pe[:, dim], label=f'dim {dim}')
    
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.title('Positional Encoding Patterns (First 8 Dimensions)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 8))
    plt.imshow(pe, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding Heatmap')
    plt.tight_layout()
    plt.show()

def test_decoder_causal_masking():
    torch.manual_seed(42)
    
    batch_size = 2
    seq_length = 5
    d_model = 512
    d_ff = 2048
    num_heads = 8
    
    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()
    
    decoder_input = torch.randn(batch_size, seq_length, d_model)
    encoder_output = torch.randn(batch_size, seq_length, d_model)
    
    attention_scores = []
    
    def attention_hook(module, input, output):
        if not attention_scores:
            scores = F.softmax(module.att_matrix, dim=-1)
            attention_scores.append(scores.detach())
    
    decoder.att.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output)
    
    att_weights = attention_scores[0]
    
    print("\nAttention Matrix Shape:", att_weights.shape)
    print("\nAttention Pattern (first head):")
    print(att_weights[0, 0].round(decimals=4))

def test_decoder_cross_attention():
    torch.manual_seed(42)
    
    batch_size = 2
    decoder_seq_len = 5
    encoder_seq_len = 7
    d_model = 512
    d_ff = 2048
    num_heads = 8
    
    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()
    
    decoder_input = torch.randn(batch_size, decoder_seq_len, d_model)
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model)
    
    cross_attention_scores = []
    
    def attention_hook(module, input, output):
        if len(cross_attention_scores) < 2:
            scores = F.softmax(module.att_matrix, dim=-1)
            cross_attention_scores.append(scores.detach())
    
    decoder.att.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output)
    
    cross_att_weights = cross_attention_scores[1]
    
    print("\nCross-Attention Matrix Shape:", cross_att_weights.shape)
    print("\nCross-Attention Pattern (first head):")
    print(cross_att_weights[0, 0].round(decimals=4))

def test_decoder_cross_attention_with_padding():
    torch.manual_seed(42)
    
    batch_size = 2
    decoder_seq_len = 5
    encoder_seq_len = 7
    d_model = 512
    d_ff = 2048
    num_heads = 8
    
    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()
    
    decoder_input = torch.randn(batch_size, decoder_seq_len, d_model)
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model)
    
    padding_mask = torch.ones(batch_size, decoder_seq_len, encoder_seq_len)
    padding_mask[:, :, -2:] = float('-inf')
    padding_mask = padding_mask.unsqueeze(1)
    
    cross_attention_scores = []
    
    def attention_hook(module, input, output):
        if len(cross_attention_scores) < 2:
            scores = F.softmax(module.att_matrix, dim=-1)
            cross_attention_scores.append(scores.detach())
    
    decoder.att.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output, padding_mask)
    
    cross_att_weights = cross_attention_scores[1]
    
    print("\nCross-Attention Matrix Shape:", cross_att_weights.shape)
    print("\nCross-Attention Pattern (first head):")
    print(cross_att_weights[0, 0].round(decimals=4))

if __name__ == '__main__':
    visualize_positional_encoding(seq_length=16, d_model=32)
    test_decoder_causal_masking()
    test_decoder_cross_attention()
    test_decoder_cross_attention_with_padding()
