from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1 