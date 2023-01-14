import numpy as np
import time
import torch
import torch.nn as nn

SOS_token = 1
EOS_token = 2

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


def train_sentence(input_tensor, target_tensor, decoder, optimizer, criterion, teacher_forcing):
    loss = 0
    optimizer.zero_grad()

    target_tensor = target_tensor.view(-1, 1)
    target_length = target_tensor.size(0)

    decoder_input = torch.tensor([[SOS_token]], device=decoder.device)
    decoder_hidden = input_tensor.view(1, 1, -1)  # drip

    if teacher_forcing:
        # teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # drip
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # use predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    optimizer.step()

    return loss.item() / target_length


def evaluate_sentence(input_tensor, target_tensor, decoder, criterion):
    loss = 0
    length = 0

    target_tensor = target_tensor.view(-1, 1)
    target_length = target_tensor.size(0)

    decoder_input = torch.tensor([[SOS_token]], device=decoder.device)
    decoder_hidden = input_tensor.view(1, 1, -1)  # drip

    with torch.no_grad():

        for di in range(target_length):
            decoder_output, decoder_hiddenn = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            length += 1
            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

        return loss.item() / length


def train(train_features, train_labels, val_features, val_labels, decoder, optimizer, criterion, epochs, teacher_forcing, learning_rate=0.01):
    log_interval = 5

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_losses = []

        print('-' * 89)
        indices = np.arange(len(train_features))
        np.random.shuffle(indices)

        for i in range(len(train_features)):
            start_time = time.time()
            loss = train_sentence(train_features[indices[i]], train_labels[indices[i]], decoder, optimizer, criterion, teacher_forcing)
            train_losses.append(loss)

            if i % log_interval == 0:
                print('| epoch {:3d} | i {:4d} | time: {:5.2f}s | train loss {:5.2f}'
                      .format(epoch, i, (time.time() - start_time), loss))

        #drip indices = np.arange(len(val_features))
        #drip np.random.shuffle(indices)
        #drip for i in range(len(val_features)):
        #drip     val_loss = evaluate_sentence(val_features[indices[i]], val_labels[indices[i]], decoder, criterion)
        #drip     val_losses.append(val_loss)

        val_losses = evaluate(val_features, val_labels, decoder, criterion)
        print('-' * 89)

        print('| epoch {:3d} | ****** | time: {:5.2f}s | train loss {:5.2f} | val loss {:5.2f} '
              .format(epoch, (time.time() - epoch_start_time), np.mean(train_losses), np.mean(val_losses)))
    print('-' * 89)


def evaluate(features, labels, decoder, criterion):
    losses = []
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    for i in range(len(features)):
        loss = evaluate_sentence(features[indices[i]], labels[indices[i]], decoder, criterion)
        losses.append(loss)
    return losses
