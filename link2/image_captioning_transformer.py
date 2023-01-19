import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder

from link2.networks import EncoderCNN


class ResidualBlock(nn.Module):
    """Represents 1D version of the residual block: https://arxiv.org/abs/1512.03385"""

    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x


class Normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super(Normalize, self).__init__()
        self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, dim=-1):
        norm = x.norm(2, dim=dim).unsqueeze(-1)
        x = self.eps * (x / norm)
        return x


class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
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
        # img_feature_channels = config["image_specs"]["img_feature_channels"]

        # Load pretrained word embeddings
        self.embedding_layer = self.embedding = nn.Embedding(vocab_size, embed_size)

        self.entry_mapping_words = nn.Linear(embed_size, d_model)
        # self.entry_mapping_img = nn.Linear(img_feature_channels, d_model)

        self.res_block = ResidualBlock(d_model)

        self.positional_encodings = PositionalEncodings(seq_len=seq_len-1, d_model=d_model, dropout=dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,  # the number of expected features in the input
            nhead=nhead,  # the number of heads in the multiheadattention models
            dim_feedforward=dim_feedforward,  # the dimension of the feedforward network model (default=2048)
            dropout=dropout
        )
        self.decoder = TransformerDecoder(decoder_layer=transformer_decoder_layer,
                                          num_layers=num_decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, features, captions,  tgt_padding_mask=None, tgt_mask=None):
        # Entry mapping for word tokens
        captions = self.embedding_layer(captions)
        captions = self.entry_mapping_words(captions)
        captions = F.leaky_relu(captions)

        captions = self.res_block(captions)
        captions = F.leaky_relu(captions)

        captions = self.positional_encodings(captions)

        # Get output from the decoder
        captions = captions.permute(1, 0, 2)
        captions = self.decoder(tgt=captions, memory=features, tgt_key_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
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

    def forward(self, images, captions):
        features = self.encoder(images)

        # Transformation for connecting encoder to the decoder
        # features = features.mean(dim=1)
        features = features.permute(1, 0, 2)

        outputs = self.decoder(features, captions)
        return outputs, 0
