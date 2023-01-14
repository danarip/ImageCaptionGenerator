import os
import numpy as np
import time
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import timm
import pickle
import image2features
import networks

# set env
cwd = os.path.dirname(os.path.abspath(__file__))
root_dir = cwd
text_dir = root_dir + "/Flickr8k_text"
dataset_dir = root_dir + "/Flickr8k_Dataset"
token_file = text_dir + "/Flickr8k.token.txt"
train_images_file = text_dir + "/Flickr_8k.trainImages.txt"
val_images_file = text_dir + "/Flickr_8k.devImages.txt"
test_images_file = text_dir + "/Flickr_8k.testImages.txt"
img2cap_file = cwd + '/data/img2cap.pkl'
train_ds_file = cwd + '/data/train_ds.pkl'
val_ds_file = cwd + '/data/validation_ds.pkl'
test_ds_file = cwd + '/data/test_ds.pkl'

# build vocab
vocab = image2features.build_vocab(train_images_file, token_file)
model = None

# train set
if not os.path.exists(train_ds_file):
    # load a pretrained model
    print("drip1")
    model = timm.create_model('xception', pretrained=True)
    model.eval()
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove the last layer
    print("drip2")
    # extract features
    train_names, train_features = image2features.extract_features(model, dataset_dir, train_images_file)
    print("drip3")
    # build dict
    train_img2cap = image2features.create_labels(train_images_file, token_file, vocab)
    print("drip4")
    # merge image name, features & encoded caption. dump to file
    train_x, train_y = image2features.create_dataset(train_names, train_features, train_img2cap)
    pickle.dump((train_x, train_y), open(train_ds_file, 'wb'))
else:
    train_x, train_y = pickle.load(open(train_ds_file, 'rb'))

# validation set
if not os.path.exists(val_ds_file):
    if model is None:
        # load a pretrained model
        model = timm.create_model('xception', pretrained=True, )
        model.eval()
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove the last layer

    # extract features
    val_names, val_features = image2features.extract_features(model, dataset_dir, val_images_file)

    # build dict
    val_img2cap = image2features.create_labels(val_images_file, token_file, vocab)

    # merge image name, features & encoded caption. dump to file
    val_x, val_y = image2features.create_dataset(val_names, val_features, val_img2cap)
    pickle.dump((val_x, val_y), open(val_ds_file, 'wb'))
else:
    val_x, val_y = pickle.load(open(val_ds_file, 'rb'))


# test set
if not os.path.exists(test_ds_file):
    if model is None:
        # load a pretrained model
        model = timm.create_model('xception', pretrained=True, )
        model.eval()
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove the last layer

    # extract features
    test_names, test_features = image2features.extract_features(model, dataset_dir, test_images_file)

    # build dict
    test_img2cap = image2features.create_labels(test_images_file, token_file, vocab)

    # merge image name, features & encoded caption. dump to file
    test_x, test_y = image2features.create_dataset(test_names, test_features, test_img2cap)
    pickle.dump((test_x, test_y), open(test_ds_file, 'wb'))
else:
    test_x, test_y = pickle.load(open(test_ds_file, 'rb'))


# hyper parameters
epochs = 10
learning_rate = 0.01
hidden_size = train_x[0].shape[1]
output_size = len(vocab)

# device configuration, as before
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create model, send it to device
decoder = networks.DecoderRNN(hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
#drip scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# train decoder
networks.train(train_x, train_y, val_x, val_y, decoder, optimizer, criterion, device, epochs, teacher_forcing=True, learning_rate=learning_rate)

# test
start_time = time.time()
test_losses = networks.evaluate(test_x, test_y, decoder, criterion)
print('| ********* | ****** | time: {:5.2f}s | test loss {:5.2f}  '
      .format((time.time() - start_time), np.mean(test_losses)))
print('-' * 89)