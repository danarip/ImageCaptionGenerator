import os
import string
import time
import numpy as np
from PIL import Image
from pickle import dump, load
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def file_read(filename):
    f = open(filename, 'r')
    text = f.read()
    f.close()
    return text


def get_img_names(filename):
    file = file_read(filename)
    images_name = file.split("\n")[:-1]
    return images_name


def build_vocab(imgs_file, caps_file):
    img_names = get_img_names(imgs_file)
    file = file_read(caps_file)

    captions = list()
    lines = file.split('\n')
    for line in lines[:-1]:
        img_name, caption = line.split('\t')
        if img_name[:-2] not in img_names:
            continue
        captions.append(caption)

    # build vocab
    train_iter = iter(captions)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>', '<sos>', '<eos>']) #drip sos/eos
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def extract_features(model, dataset_dir, imgs_file, img_size=(299, 299), batch_size=2):
    img_names = get_img_names(imgs_file)

    # load and preprocess image
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()])

    img_features = list()

    start_time = time.time()
    for i, img in enumerate(img_names):
        filename = dataset_dir + "/" + img
        image = Image.open(filename).convert('RGB')
        tensor = transform(image).unsqueeze(0)  # transform and add batch dimension

        # model prediction
        with torch.no_grad():
            score = model(tensor)
            features = torch.nn.functional.softmax(score[0], dim=0)

        img_features.append(features)

        if i % 100 == 0:
            print('{:3d} | extract_features time: {:5.2f}s'.format(i, (time.time() - start_time)))

    return img_names, img_features


def create_labels(imgs_file, caps_file, vocab):
    img_names = get_img_names(imgs_file)
    file = file_read(caps_file)
    stoi = vocab.vocab.get_stoi()
    itos = vocab.vocab.get_itos()

    img2cap = dict()
    lines = file.split('\n')
    for line in lines[:-1]:
        img_name, caption = line.split('\t')
        if img_name[:-2] not in img_names:
            continue

        #drip encoded_caption = [stoi[token] for token in caption.lower().split(" ")]
        encoded_caption = vocab.vocab.lookup_indices(caption.lower().split(" "))
        # add sos/eos markers
        encoded_caption = [stoi['<sos>']] + encoded_caption + [stoi['<eos>']]
        # dbg print(f'[{img_name[:-2]}] -> {[itos[token] for token in encoded_caption]}')

        if img_name[:-2] not in img2cap:
            img2cap[img_name[:-2]] = [encoded_caption]
        else:
            img2cap[img_name[:-2]].append(encoded_caption)
    return img2cap


def create_dataset(names, features, img2cap):
    x = list()
    y = list()
    for i, name in enumerate(names):
        for j in range(5):
            x.append(features[i].view(1, -1))
            y.append(torch.tensor(img2cap[name][j], dtype=int).view(1, -1))
    return x, y
