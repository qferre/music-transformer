import torch
import math
from torch import nn, Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, input_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2) * (-math.log(10000.0) / input_dim))
        pe = torch.zeros(max_len, 1, input_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, input_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag. 
    Used to make a causal mask in self-attention and prevent the flow of 
    information from past to future tokens.
    """
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



## Model Definition

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()

        # Module definitions

        self.pos_encoder = PositionalEncoding(input_dim=d_model, dropout=dropout)

        encoder_layer_template = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer_template, nlayers)
        # NOTE: self.transformer_encoder expects the input to have a number of features equal to d_model

        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
       
        # Try to produce one note at a time, so the output dimension should just 
        # be ntoken, since the output should be a one-hot encoding of the produced note
        self.decoder = nn.Linear(
            in_features = d_model,
            out_features = ntoken
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [1, batch_size, ntoken]
        """

        # Embedding
        src = self.encoder(src) * math.sqrt(self.d_model)
        # NOTE: After this stage, src should have size [seq_len, batch_size, d_model]
        # This means you can modify it by removing encoder and passing directly
        # a tensor of such shape to self.pos_encoder if you want to use a
        # custom embedding.

        # Add positional encoding
        src = self.pos_encoder(src)

        # Pass through the Transformer blocks
        output = self.transformer_encoder(src, src_mask)

        # NOTE: At this point, output should have size [seq_len, batch_size, d_model]

        # Pool across the seq_len dimension
        output = torch.mean(output, dim=0)

       
        # Decode and produce output
        output = self.decoder(output)
        return output

