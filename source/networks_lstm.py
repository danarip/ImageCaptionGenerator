import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, device):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]  # 2048 x 7 x 7
        self.resnet = nn.Sequential(*modules)
        self.device = device  # drip: can be removed?
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. EncoderCNN={self.num_parameters:,}")

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,49,2048)

        return features


# Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, device):
        super(Attention, self).__init__()
        self.device = device
        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. Attention={self.num_parameters:,}")

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,num_layers,attemtion_dim)

        attention_scores = self.A(combined_states)  # (batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,num_layers)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,num_layers)

        return alpha, attention_weights


# Attention Decoder
class DecoderLSTMAttention(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob=0.3):
        super().__init__()
        self.device = device

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim, device=device)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. DecoderLSTMAttention={self.num_parameters:,}")

    def forward(self, features, captions, seq_len):

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # h, c=1024x512=(batch_size, decoder_dim)

        # get the seq length to iterate
        batch_size = captions.shape[0]
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_len - 1, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_len - 1, num_features).to(self.device)

        for s in range(seq_len-1):
            alpha, context = self.attention(features,
                                            h)  # features: 1024x49x2048, h: 1024x512, context:1024x2048, alpha:1024x49
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))  # 1024 x 2994 = batch x vocab_size

            preds[:, s] = output  # 1024 x 29 x 2994 = batch x (seq_len-1) x vocab_size
            alphas[:, s] = alpha

        return preds, alphas

    def generate_captions_greedy_lstm(self, features, max_len, vocab):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        # starting input
        predicted_word_idx = (vocab.stoi['<SOS>'] * torch.ones(size=(batch_size,), dtype=torch.long)).to(self.device)
        embeds = self.embedding(predicted_word_idx)

        captions = torch.zeros(size=(batch_size, max_len - 1), dtype=torch.long).to(features.device)
        captions_prob = torch.zeros(size=(batch_size, max_len - 1, vocab.size()), dtype=torch.float).to(features.device)

        for i in range(max_len-1):
            alpha, context = self.attention(features, h)

            # store the alpha score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds, context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output_prob = self.fcn(self.drop(h))
            output_prob = output_prob.view(batch_size, -1)

            # select the word with most val
            predicted_word_idx = output_prob.argmax(dim=1)

            # save the generated word
            captions[:, i] = predicted_word_idx
            captions_prob[:, i, :] = output_prob

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx)

        # covert the vocab idx to words and return sentence
        return captions, alphas, captions_prob

    def init_hidden_state(self,
                          encoder_out  # 1024 x 49 x 2048 <> Batch x square_resnet x rep length
                          ):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


# In[28]:


class EncoderDecoderLSTMAttention(nn.Module):
    def __init__(self,
                 embed_size,
                 vocab_size,
                 attention_dim,
                 encoder_dim,
                 decoder_dim,
                 device):
        super().__init__()
        self.device = device
        self.encoder = EncoderCNN(device=device)
        self.decoder = DecoderLSTMAttention(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            device=device
        )
        self.num_parameters = pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"no. EncoderDecoderLSTMAttention={self.num_parameters:,}")

    def forward(self, images, captions, seq_len):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, seq_len)
        return outputs
