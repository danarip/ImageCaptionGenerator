'''
Nice Transformer Image Caption Diagram:
https://heartbeat.comet.ml/caption-your-images-with-a-cnn-transformer-hybrid-model-a980f437da7b
'''
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from root import cwd
from link2.data_preprocessing import FlickrDataset, CapsCollate
from link2.utils import show_image
from link2.networks_transformer import EncoderDecoderTransformer

# location of the training data
data_location = f"{cwd}/data/flickr8k"
id_run = datetime.now().strftime("%Y%m%d_%H%M%S")
tb = SummaryWriter(log_dir=cwd + "/tensorboard/link2/images_" + id_run)


# helper function to save the model
def save_model(model2save, num_epochs, id_for_save):
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': embed_size,
        'vocab_size': len(dataset.vocab),
        'attention_dim': attention_dim,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'state_dict': model2save.state_dict()
    }
    torch.save(model_state, f"{cwd}/models/attention_model_state_{id_for_save}.pth")


# Initiate the Dataset and Dataloader
# setting the constants
BATCH_SIZE = 1024
NUM_WORKER = 4

# defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# testing the dataset class
dataset = FlickrDataset(root_dir=data_location + "/Images/", captions_file=data_location + "/captions.txt",
                        transform=transforms)

# writing the dataloader
pad_idx = dataset.vocab.stoi["<PAD>"]
data_loader = DataLoader(dataset=dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKER,
                         shuffle=True,
                         collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 3) Defining the Model Architecture
"""
Model is seq2seq model. 
In the **encoder** pretrained ResNet model is used to extract the features. 
Decoder, is the implementation of the Bahdanau Attention Decoder. 
In the decoder model **LSTM cell**.
"""

# Hyperparams
embed_size = 256
vocab_size = len(dataset.vocab)
attention_dim = 256
encoder_dim = 2048
decoder_dim = 256
learning_rate = 3e-4
seq_len = 20
num_decoder_layers = 2
image_dimension = 2048
nhead = 4
d_model = 2048
dim_feedforward = 2048
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

model = model.to(device)
# model = nn.DataParallel(model, device_ids=[0, 1]).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5) Training Job from above configs
num_epochs = 100
print_every = 5
num_batches = len(data_loader)
num_of_pics_to_show = 3

time_start_training = time.time()
for epoch in range(num_epochs):
    time_start_epoch = time.time()
    for idx, (image, captions) in enumerate(data_loader):
        image, captions = image.to(device), captions.to(device)
        print(f"{captions.shape[1]}")
        captions = captions[:, :seq_len-1]
        print(f"{captions.shape[1]}")

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)
        outputs = outputs[:, :-1]

        # Calculate the batch loss.
        targets = captions[:, 1:]
        loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))

        # Backward pass.
        loss.backward()
        tb.add_scalar("loss", loss.item(), epoch + idx / num_batches)

        # Update the parameters in the optimizer.
        optimizer.step()

        if idx % print_every == 0:
            s = f"E{epoch:03d}/{num_epochs:03d} i{idx:03d}/{num_batches:03d} loss{loss.item():.5f}"
            print(s)

            # generate the caption
            model.eval()
            with torch.no_grad():
                for _ in range(num_of_pics_to_show):
                    dataiter = iter(data_loader)
                    img, pred_caption = next(dataiter)
                    caption = ' '.join(caps)
                    show_image(img[0], title=s + ":" + caption, tb=tb)

            model.train()
    time_from_training_start = (time.time() - time_start_training)
    expected_time = num_epochs / (epoch + 1) * time_from_training_start
    print(f"epoch time {time_from_training_start / (epoch + 1):.2f} [sec], elapsed_time {time_from_training_start}")
    print(f"expected total time {expected_time:.2f}")

    # save the latest model
    save_model(model, epoch, f"{id_run}_{epoch:03d}")

