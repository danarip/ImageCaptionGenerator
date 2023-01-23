import math

import torch
import torch.nn as nn
from torch.nn import TransformerDecoderLayer, TransformerDecoder

from source.networks_lstm import EncoderCNN


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. PositionalEncoding={self.num_parameters}")

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        return x


class CaptionDecoder(nn.Module):
    """Decoder for image captions.

    Generates prediction for next caption word given the prviously
    generated word and image features extracted from CNN.
    """

    def __init__(self,
                 image_dimension,
                 embed_size,
                 vocab_size,
                 seq_len,
                 num_decoder_layers,  # TransformerDecoder: the number of sub-decoder-layers in the decoder
                 d_model,  # TransformerDecoderLayer: the number of expected features in the input
                 nhead,  # TransformerDecoderLayer: the number of heads in the multiheadattention models
                 dim_feedforward,
                 # TransformerDecoderLayer: the dimension of the feedforward network model (default=2048).
                 device,
                 dropout
                 ):
        """Initializes the model."""
        super(CaptionDecoder, self).__init__()
        self.image_dimension = image_dimension
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_decoder_layers = num_decoder_layers
        self.nhead = nhead
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = device

        # Load pretrained word embeddings
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.encoder_to_decoder = nn.Linear(self.image_dimension, self.d_model)

        self.positional_encodings = PositionalEncoding(d_model, dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,  # the number of expected features in the input
            nhead=nhead,  # the number of heads in the multiheadattention models
            dim_feedforward=dim_feedforward,  # the dimension of the feedforward network model (default=2048)
            dropout=dropout
        )
        self.decoder = TransformerDecoder(decoder_layer=transformer_decoder_layer,
                                          num_layers=num_decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. CaptionDecoder={self.num_parameters}")

    def forward(self, features, captions,  tgt_key_padding_mask=None, tgt_mask=None):
        # Entry mapping for word tokens
        captions = self.embedding_layer(captions) * math.sqrt(self.d_model)
        captions = self.positional_encodings(captions)
        features = self.encoder_to_decoder(features)

        # Get output from the decoder
        captions = captions.permute(1, 0, 2)
        captions = self.decoder(tgt=captions, memory=features, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
        captions = captions.permute(1, 0, 2)

        captions = self.classifier(captions)
        return captions


class EncoderDecoderTransformer(nn.Module):
    def __init__(self,
                 image_dimension,
                 embed_size,
                 vocab_size,
                 seq_len,
                 num_decoder_layers,  # TransformerDecoder: the number of sub-decoder-layers in the decoder
                 d_model,  # TransformerDecoderLayer: the number of expected features in the input
                 nhead,  # TransformerDecoderLayer: the number of heads in the multiheadattention models
                 dim_feedforward,
                 # TransformerDecoderLayer: the dimension of the feedforward network model (default=2048).
                 device,
                 dropout
                 ):
        super().__init__()
        self.device = device
        self.encoder = EncoderCNN(device=device)
        self.decoder = CaptionDecoder(
            image_dimension=image_dimension,
            embed_size=embed_size,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_decoder_layers=num_decoder_layers,  # TransformerDecoder: the number of sub-decoder-layers in the decoder
            d_model=d_model,  # TransformerDecoderLayer: the number of expected features in the input
            nhead=nhead,  # TransformerDecoderLayer: the number of heads in the multiheadattention models
            dim_feedforward=dim_feedforward, # TransformerDecoderLayer: the dimension of the feedforward network model (default=2048).
            device=device,
            dropout=dropout
        )
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. EncoderDecoderTransformer={self.num_parameters}")


    def forward(self, images, captions, tgt_key_padding_mask=None, tgt_mask=None):
        features = self.encoder(images)

        # Transformation for connecting encoder to the decoder
        # features = features.mean(dim=1)
        features = features.permute(1, 0, 2)

        outputs = self.decoder(features, captions, tgt_key_padding_mask, tgt_mask)
        return outputs
