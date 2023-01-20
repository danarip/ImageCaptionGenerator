'''
Taken from: https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k/notebook
and from: https://www.kaggle.com/code/mdteach/image-captioning-with-attention-pytorch

Attention type: Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015)
By: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: https://arxiv.org/abs/1409.0473
Blog to explain the attention mechanism: https://machinelearningmastery.com/the-bahdanau-attention-mechanism

CNN-SamLynnEvans Architecture
https://github.com/senadkurtisi/pytorch-image-captioning
'''
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from root import cwd
from link2.data_preprocessing import FlickrDataset, CapsCollate
from link2.utils import show_image
from link2.networks_transformer import EncoderDecoderTransformer
from link2.data_preprocessing import transforms
from link2.decoding_utils import greedy_decoding

# locations of the training / validation data
data_train_images_path = f"{cwd}/data/flickr8k/Flickr8kTrainImages/"
data_train_captions = f"{cwd}/data/flickr8k/captions_train.txt"
data_validation_images_path = f"{cwd}/data/flickr8k/Flickr8kTrainImages/"
data_validation_captions = f"{cwd}/data/flickr8k/captions_train.txt"

id_run = datetime.now().strftime("%Y%m%d_%H%M%S")
tb = SummaryWriter(log_dir=cwd + "/tensorboard/link3/transformers_images_" + id_run)

# Initiate the Dataset and Dataloader
# setting the constants
BATCH_SIZE = 256
BATCH_SIZE_VAL = 10
NUM_WORKER = 4
seq_len = 30

# Train dataset and dataloader
dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions, transform=transforms)
pad_idx = dataset_train.vocab.stoi["<PAD>"]
sos_idx = dataset_train.vocab.stoi["<SOS>"]
eos_idx = dataset_train.vocab.stoi["<EOS>"]
idx2word = dataset_train.vocab.itos
data_loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True,
                               collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                      max_len=seq_len))

# Validation dataset and dataloader
dataset_validation = FlickrDataset(root_dir=data_validation_images_path, captions_file=data_validation_captions,
                                   transform=transforms, vocab=dataset_train.vocab)
data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=BATCH_SIZE, num_workers=NUM_WORKER,
                                    shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                                         max_len=seq_len))

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"device to rune on {device}")

# 3) Defining the Model Architecture
"""
Model is seq2seq model. 
In the **encoder** pretrained ResNet model is used to extract the features. 
Decoder, is the implementation of the Bahdanau Attention Decoder. 
In the decoder model **LSTM cell**.
"""

# Hyperparams
embed_size = 256
vocab_size = len(dataset_train.vocab)
attention_dim = 256
encoder_dim = 256
decoder_dim = 256
learning_rate = 1e-4
num_decoder_layers = 4
image_dimension = 2048
nhead = 8
d_model = 2048
dim_feedforward = 512
dropout = 0.3

model = EncoderDecoderTransformer(
    image_dimension=image_dimension,
    embed_size=embed_size,
    vocab_size=vocab_size,
    seq_len=seq_len,
    num_decoder_layers=num_decoder_layers,  # TransformerDecoder: the number of sub-decoder-layers in the decoder
    d_model=d_model,  # TransformerDecoderLayer: the number of expected features in the input
    nhead=nhead,  # TransformerDecoderLayer: the number of heads in the multiheadattention models
    dim_feedforward=dim_feedforward,
    # TransformerDecoderLayer: the dimension of the feedforward network model (default=2048).
    device=device,
    dropout=dropout
)
"""
model = EncoderDecoderTransformer(
    "resnet34",
    226,
    vocab_size=len(dataset_train.vocab),
    max_seq_length=seq_len,
    num_decoder_layers=6,
    n_head=8,
    d_model=512,
    fc_dim=512,
    dropout=0.2)
"""
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset_train.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5) Training Job from above configs
num_epochs = 100
print_every = 5
save_every_epochs = 10
num_batches = len(data_loader_train)
num_of_pics_to_show = 3

tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len-1).to(device) != 0

time_start_training = time.time()
for epoch in range(num_epochs):
    time_start_epoch = time.time()
    for idx, (image, captions) in enumerate(data_loader_train):
        image, captions = image.to(device), captions.to(device)
        captions = captions[:, 1:]
        tgt_key_padding_mask = captions != pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
        tgt_key_padding_mask.to(device)
        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs = model(image, captions, tgt_mask=tgt_mask)
        outputs = outputs[:, :-1]

        # Calculate the batch loss.
        targets = captions[:, 1:]
        loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))

        # Backward pass.
        loss.backward()
        time_of_loss = epoch + idx / num_batches
        print(f"time of loss={time_of_loss}")
        tb.add_scalar("loss", loss.item(), time_of_loss)

        # Update the parameters in the optimizer.
        optimizer.step()

        if idx % print_every == 0:
            s = f"E{epoch:03d}/{num_epochs:03d} i{idx:03d}/{num_batches:03d} loss{loss.item():.5f}"
            print(s)

            # generate the caption
            model.eval()
            with torch.no_grad():
                dataiter = iter(data_loader_validation)
                img, pred_caption = next(dataiter)
                img, pred_caption = img.to(device), pred_caption.to(device)
                # greedy_decoding(model, img_features_batched, sos_id, eos_id, pad_id, idx2word, max_len, device):
                captions_pred_batch = greedy_decoding(model, img, sos_idx, eos_idx, pad_idx, idx2word,
                                                      max_len=seq_len - 1, device=device, tgt_mask=tgt_mask)
                captions_pred_batch = captions_pred_batch[:BATCH_SIZE_VAL]

                for i, caption in enumerate(captions_pred_batch):
                    caption = ' '.join(caption)
                    show_image(img[i], title=s + ":" + caption, tb=tb)

            model.train()
    time_from_training_start = (time.time() - time_start_training)
    expected_time = num_epochs / (epoch + 1) * time_from_training_start
    print(f"epoch time {time_from_training_start / (epoch + 1):.2f} [sec], elapsed_time {time_from_training_start}")
    print(f"expected total time {expected_time:.2f}")

    # save the latest model
    # save the latest model
    if epoch % save_every_epochs == 0 or epoch == num_epochs - 1:
        torch.save(model, f"{cwd}/models/attention_model_state_{id_run}_{epoch:03d}.pth")
