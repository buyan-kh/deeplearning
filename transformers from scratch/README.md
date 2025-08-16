# Transformer

A PyTorch implementation of the Transformer model.

## Running Tests

To run the tests, execute:

```bash
python tests.py
```

This will run a series of tests for the different components of the Transformer model.

```bash
python3 tests.py
tokenizer_config.json: 100%|███████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 74.0kB/s]
config.json: 100%|███████████████████████████████████████████████████████████| 629/629 [00:00<00:00, 2.47MB/s]
vocab.txt: 100%|███████████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 1.82MB/s]
Tokenizer vocabulary size: 30522
Input shape: torch.Size([2, 16])
Embedded shape after projection: torch.Size([2, 16, 768])

Output Statistics:
Mean: 0.0000
Std: 1.0000
Min: -2.7968
Max: 2.8519

Attention Analysis:
Unmasked positions mean: 0.8078
Masked positions mean: 0.8078

Is the masking working? Yes

All tests passed successfully!

Output Statistics:
Mean: 0.0000
Std: 1.0000
Min: -4.3617
Max: 4.5787

Shape Analysis:
Input shape: torch.Size([32, 20, 512])
Output shape: torch.Size([32, 20, 512])
Expected shape matches: Yes

All tests passed successfully!

Final Output Statistics:
Mean: 0.0000
Std: 1.0000
Min: -3.7172
Max: 4.1310

All tests passed successfully!

Output Analysis:
Output shape: torch.Size([2, 17, 30522])

Training Check:
Loss value: 10.7329
Has gradients: True
Buyantogtokhs-MacBook-Pro:transformers buyantogtokh$
```
