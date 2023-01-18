'''
Taken from: https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k/notebook
and from: https://www.kaggle.com/code/mdteach/image-captioning-with-attention-pytorch

Attention type: Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015)
By: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: https://arxiv.org/abs/1409.0473
Blog to explain the attention mechanism: https://machinelearningmastery.com/the-bahdanau-attention-mechanism

CNN-Transformer Architecture
https://github.com/senadkurtisi/pytorch-image-captioning
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
from link2.utils import show_image, plot_attention
from link2.networks import EncoderDecoder

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
embed_size = 300
vocab_size = len(dataset.vocab)
attention_dim = 256
encoder_dim = 2048
decoder_dim = 512
learning_rate = 3e-4

# In[30]:


# init model
model = EncoderDecoder(
    embed_size=300,
    vocab_size=len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512,
    device=device
)

model = model.to(device)
#model = nn.DataParallel(model, device_ids=[0, 1]).to(device)

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

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)

        # Calculate the batch loss.
        targets = captions[:, 1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

        # Backward pass.
        loss.backward()
        tb.add_scalar("loss", loss.item(), epoch+idx/num_batches)

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
                    img, _ = next(dataiter)
                    features = model.module.encoder(img[0:1].to(device))  # drip: added module for parallelization
                    caps, alphas = model.module.decoder.generate_caption(features, vocab=dataset.vocab)  # drip: added module for parallelization
                    caption = ' '.join(caps)
                    show_image(img[0], title=s + ":" + caption, tb=tb)

            model.train()
    time_from_training_start = (time.time() - time_start_training)
    expected_time = num_epochs / (epoch + 1) * time_from_training_start
    print(f"epoch time {time_from_training_start / (epoch + 1):.2f} [sec], elapsed_time {time_from_training_start}")
    print(f"expected total time {expected_time:.2f}")

    # save the latest model
    save_model(model, epoch, f"{id_run}_{epoch:03d}")


# ## 6 Visualizing the attentions
# Defining helper functions
# Given the image generate captions and attention scores. Plot the attention scores in the image

# generate caption
def get_caps_from(features_tensors):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0], title=caption)

    return caps, alphas


# Show attention


# show any 1
dataiter = iter(data_loader)
images, _ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps, alphas = get_caps_from(img.unsqueeze(0))

plot_attention(img1, caps, alphas)

# In[ ]:


# show any 1
dataiter = iter(data_loader)
images, _ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps, alphas = get_caps_from(img.unsqueeze(0))

plot_attention(img1, caps, alphas)

# In[ ]:


# show any 1
dataiter = iter(data_loader)
images, _ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps, alphas = get_caps_from(img.unsqueeze(0))

plot_attention(img1, caps, alphas)

# In[ ]:


# show any 1
dataiter = iter(data_loader)
images, _ = next(dataiter)

img = images[0].detach().clone()
img1 = images[0].detach().clone()
caps, alphas = get_caps_from(img.unsqueeze(0))

plot_attention(img1, caps, alphas)
