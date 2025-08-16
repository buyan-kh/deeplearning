import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from embedding import EmbeddingWithProjection
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from encoder_decoder import TransformerEncoderDecoder
from transformer import Transformer

def test_embedding_implementation():
    from transformers import AutoTokenizer
    import torch

    d_model = 768
    d_embed = 1024
    vocab_size = 30522

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, use_multiprocessing=False)
    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    max_position_embeddings = 512
    model_inputs = tokenizer(sequences, truncation=True, padding="longest")

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocabulary size: {vocab_size}")

    input_tensor = torch.tensor(model_inputs['input_ids'])
    embedder = EmbeddingWithProjection(vocab_size=vocab_size, d_embed=d_embed, d_model=d_model)
    output = embedder(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Embedded shape after projection: {output.shape}")

def test_transformer_encoder():
    torch.manual_seed(42)
    
    batch_size = 32
    seq_length = 20
    d_model = 512
    d_ff = 2048
    num_heads = 8
    
    encoder = TransformerEncoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    encoder.eval()
    
    input_sequence = torch.ones(batch_size, seq_length, d_model)
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[:, 15:] = 0
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)
    
    attention_patterns = []
    
    def attention_hook(module, input, output):
        attention_patterns.append(output)
    
    encoder.att.register_forward_hook(attention_hook)
    
    with torch.no_grad():
        output = encoder(input_sequence, attention_mask)
    
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("\nOutput Statistics:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")
    
    if attention_patterns:
        unmasked_attention = output[:, :15, :].abs().mean()
        masked_attention = output[:, 15:, :].abs().mean()
        
        print("\nAttention Analysis:")
        print(f"Unmasked positions mean: {unmasked_attention:.4f}")
        print(f"Masked positions mean: {masked_attention:.4f}")
        print("\nIs the masking working?", "Yes" if unmasked_attention != masked_attention else "No")
    
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
    print("\nAll tests passed successfully!")

def test_transformer_decoder():
    torch.manual_seed(42)
    
    batch_size = 32
    seq_length = 20
    encoder_seq_length = 22
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
    encoder_output = torch.randn(batch_size, encoder_seq_length, d_model)
    
    padding_mask = torch.ones(batch_size, seq_length, encoder_seq_length)
    padding_mask[:, :, 18:] = 0
    padding_mask = padding_mask.unsqueeze(1)
    
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output, padding_mask)
    
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("\nOutput Statistics:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")
    
    print("\nShape Analysis:")
    print(f"Input shape: {decoder_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape matches: {'Yes' if decoder_input.shape == output.shape else 'No'}")
    
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
    print("\nAll tests passed successfully!")

def test_transformer_encoder_decoder_stack():
    torch.manual_seed(42)
    
    batch_size = 8
    seq_length = 10
    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_layers = 6
    
    transformer = TransformerEncoderDecoder(
        num_layer=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    transformer.eval()
    
    encoder_input = torch.randn(batch_size, seq_length, d_model)
    decoder_input = torch.randn(batch_size, seq_length, d_model)
    
    padding_mask = torch.ones(batch_size, seq_length)
    padding_mask[:, -2:] = 0
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        output = transformer(encoder_input, decoder_input, padding_mask)
    
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("\nFinal Output Statistics:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")
    
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
    print("\nAll tests passed successfully!")

def test_complete_transformer():
    d_model = 768
    d_embed = 1024
    d_ff = 2048
    num_heads = 8
    num_layers = 6
    max_position_embeddings = 512
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_fast=True, use_multiprocessing=False)
    vocab_size = tokenizer.vocab_size
    
    src_sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
    tgt_sequences = ["J'ai attendu un cours HuggingFace toute ma vie.", "Moi aussi!"]
    
    src_inputs = tokenizer(src_sequences, truncation=True, padding="longest", return_tensors="pt")
    tgt_inputs = tokenizer(tgt_sequences, truncation=True, padding="longest", return_tensors="pt")
    
    transformer = Transformer(
        num_layer=num_layers,
        d_model=d_model,
        d_embed=d_embed,
        d_ff=d_ff,
        num_head=num_heads,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings
    )
    transformer.eval()
    
    padding_mask = src_inputs['attention_mask'].unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        output = transformer(
            src_tokens=src_inputs['input_ids'],
            tgt_tokens=tgt_inputs['input_ids'],
            padding_mask=padding_mask
        )
    
    print("\nOutput Analysis:")
    print(f"Output shape: {output.shape}")
    
    transformer.train()
    output = transformer(
        src_tokens=src_inputs['input_ids'],
        tgt_tokens=tgt_inputs['input_ids'],
        padding_mask=padding_mask
    )
    
    loss = F.nll_loss(
        output.view(-1, vocab_size),
        tgt_inputs['input_ids'].view(-1)
    )
    loss.backward()
    
    has_gradients = all(p.grad is not None for p in transformer.parameters())
    print("\nTraining Check:")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Has gradients: {has_gradients}")

if __name__ == '__main__':
    test_embedding_implementation()
    test_transformer_encoder()
    test_transformer_decoder()
    test_transformer_encoder_decoder_stack()
    test_complete_transformer()
