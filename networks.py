import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SOS_token = 1
EOS_token = 2


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
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


class DecoderRNN2(nn.Module):
    def __init__(self,
                 feature_size,  # 2048
                 hidden_size,  # we decide
                 output_size,  # Vocabulary size
                 num_layers=1,  # num layers
                 ):
        super(DecoderRNN2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input = nn.Linear(feature_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=self.num_layers, dropout=0.5)  # drip
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, labels, features, init):
        batch_size = labels.shape[0]
        if init:
            hidden = self.input(features)
        else:
            hidden = features
        hidden = hidden.view(self.num_layers, batch_size, -1)  # 1 x batch size x hidden size
        output = self.embedding(labels)  # batch size x seq length x representation size
        output = nn.functional.relu(output)
        output = torch.transpose(output, 0, 1)  # seq length x batch size x representation size
        output, hidden = self.gru(output, hidden)
        # output: seq length x batch size x hidden size
        output = self.output_layer(output)  #
        output = self.logsoftmax(output)
        return output, hidden


def train_sentence(input_tensor, target_tensor, decoder, optimizer, criterion, device, teacher_forcing, max_length):
    loss = 0
    optimizer.zero_grad()
    batch_size = input_tensor.shape[0]

    # target_tensor = target_tensor.view(-1, 1).to(device)
    # target_length = target_tensor.size(0)

    # decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = input_tensor.view(1, batch_size, -1).to(device)  # drip

    if teacher_forcing:
        # teacher forcing: Feed the target as the next input
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  # drip
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # use predictions as the next input
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    optimizer.step()

    return loss.item() / max_length


def train_sentences_teacher_forcing(input_tensor, labels_tensor, decoder, optimizer, criterion, device):
    loss = 0
    optimizer.zero_grad()
    batch_size = input_tensor.shape[0]
    max_length = labels_tensor.shape[1]

    decoder_hidden = input_tensor.view(1, batch_size, -1).to(device)  # drip
    decoder_input = labels_tensor[:, :-1]

    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, init=True)
    decoder_output = torch.permute(decoder_output, (1, 2, 0))  # drip
    loss += criterion(decoder_output, labels_tensor[:, 1:])

    loss.backward()
    optimizer.step()

    return loss.item() / (batch_size * (max_length - 1))


def train_sentences_non_teacher_forcing(input_tensor, labels_tensor, decoder, optimizer, criterion, device, max_length):
    loss = 0
    optimizer.zero_grad()
    batch_size = input_tensor.shape[0]

    decoder_hidden = input_tensor.view(1, batch_size, -1).to(device)  # drip
    decoder_input = labels_tensor[:, 0:1]

    for di in range(max_length - 1):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, init=di == 0)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze(0).detach()  # detach from history as input

        decoder_output = torch.permute(decoder_output, (1, 2, 0))  # drip
        crit = criterion(decoder_output, labels_tensor[:, (di + 1):(di + 2)])
        loss += crit

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_sentences(input_tensor, labels_tensor, decoder, criterion, device, max_length):
    with torch.no_grad():
        loss = 0
        batch_size = input_tensor.shape[0]

        decoder_hidden = input_tensor.view(1, batch_size, -1).to(device)  # drip
        decoder_input = labels_tensor[:, 0:1].to(device)

        for di in range(max_length - 1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, init=di == 0)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(0).detach()  # detach from history as input

            decoder_output = torch.permute(decoder_output, (1, 2, 0))  # drip
            crit = criterion(decoder_output, labels_tensor[:, (di + 1):(di + 2)])
            loss += crit

    return loss.item()


def evaluate_sentence(input_tensor, target_tensor, decoder, criterion, device):
    loss = 0
    length = 0

    target_tensor = target_tensor.view(-1, 1).to(device)
    target_length = target_tensor.size(0)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = input_tensor.view(1, 1, -1).to(device)  # drip

    with torch.no_grad():

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            length += 1
            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:
                break

        return loss.item()  # / length


def train(train_features, train_labels, val_features, val_labels, decoder, optimizer, scheduler, criterion,
          device, epochs, batch_size, summary_writer):
    log_interval = 10

    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # val_dataset = TensorDataset(val_features, val_labels)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_losses = []

        print('-' * 89)
        indices = np.arange(len(train_features))
        np.random.shuffle(indices)

        # for i in range(len(train_features)):
        for i, (features, labels) in enumerate(train_dataloader):
            # send data to device
            features = features.to(device)
            labels = labels.to(device)

            start_time = time.time()
            # loss = train_sentence(train_features[indices[i]], train_labels[indices[i]], decoder, optimizer, criterion, device, teacher_forcing)
            loss = train_sentences_non_teacher_forcing(features, labels, decoder, optimizer, criterion, device,
                                                       max_length=20)
            train_losses.append(loss)

            if i % log_interval == 0:
                print('| epoch {:3d} | i {:4d} | samples: {:6d} | time: {:5.2f}s | train loss {:5.5f}'
                      .format(epoch, i, i * batch_size, (time.time() - start_time), loss))

        #scheduler.step()


        # val_losses = evaluate(val_features, val_labels, decoder, criterion)
        epoch_val_loss = evaluate_sentences(val_features, val_labels, decoder, criterion, device, max_length=20)

        epoch_train_loss = np.mean(train_losses)
        scheduler.step(np.mean(epoch_train_loss))

        print('-' * 89)
        print('| epoch {:3d} | ****** | time: {:5.2f}s | train loss {:5.5f} | val loss {:5.5f} '
              .format(epoch, (time.time() - epoch_start_time), epoch_train_loss, epoch_val_loss))
        summary_writer.add_scalars("losses", {"train": epoch_train_loss,
                                              "val": epoch_val_loss}, epoch)
    print('-' * 89)


def evaluate(features, labels, decoder, criterion, device, max_length):
    labels = labels.to(device)
    loss = evaluate_sentences(features, labels, decoder, criterion, device, max_length=max_length)
    return loss
