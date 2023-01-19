import torch
from torch import nn
from link2.networks import EncoderCNN


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        max_seq_length,
        num_decoder_layers,
        n_head,
        decoder_dim,
        device,
        dropout=0.3
    ):
        # def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob=0.3):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.device = device

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_seq_length, embed_size)
        decoder_layer = nn.TransformerDecoderLayer(embed_size, n_head, decoder_dim, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, encoder_out, caption, target_pad_mask):
        batch_size = caption.shape[0]
        sequence_length = caption.shape[1]

        # Positional embedding at the decoder
        scale = torch.sqrt(torch.tensor([self.embed_size])).to(self.device)
        x = self.token_embedding(caption) * scale
        position = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        x += self.positional_embedding(position)

        # Image feature at the encoder
        mean_encoder_out = encoder_out.mean(dim=1)

        # Decoder
        encoder_memory = self.encoder_linear(mean_encoder_out)
        encoder_memory = encoder_memory.unsqueeze(1)  # .permute(1, 0, 2)
        target_subsequent_mask = (
            nn.Transformer()
            .generate_square_subsequent_mask(x.shape[1])
            .to(self.device)
        )
        x = self.transformer_decoder(
            x,
            encoder_memory,
            tgt_mask=target_subsequent_mask,
            tgt_key_padding_mask=target_pad_mask,
        )
        out = self.linear(x)
        return out, None


class EncoderDecoderTransformer(nn.Module):
    def __init__(self,
                 embed_size,
                 vocab_size,
                 max_seq_length,
                 num_decoder_layers,
                 n_head,
                 decoder_dim,
                 device,
                 drop_prob=0.3):
        super().__init__()
        self.device = device
        self.encoder = EncoderCNN(device=device)
        self.decoder = DecoderTransformer(
            embed_size=embed_size,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            num_decoder_layers=num_decoder_layers,
            n_head=n_head,
            decoder_dim=decoder_dim,
            device=device,
            dropout=0.3,
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


