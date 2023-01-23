import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T

import spacy

spacy_eng = spacy.load("en_core_web_sm")

# defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transforms_advanced = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    T.RandomAffine(10),
    T.RandomGrayscale(0.05),
    T.RandomHorizontalFlip(0.05),
    T.RandomVerticalFlip(0.05),
    T.GaussianBlur(5),
    T.RandomErasing(0.05)
])


class Vocabulary:
    def __init__(self, freq_threshold):
        # setting the pre-reserved tokens int to string tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """

    def __init__(self,
                 root_dir,
                 captions_file,
                 transform=None,
                 freq_threshold=5,
                 vocab=None,
                 data_limit=None,
                 do_augmentation=False,
                 augmentation_probability=0.2):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.random = np.random.RandomState()
        self.do_augmentation = do_augmentation
        self.augmentation_probability = augmentation_probability

        # Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        # If needed truncating the data for faster running
        if data_limit is not None:
            self.imgs = self.imgs[:data_limit]
            self.captions = self.captions[:data_limit]

        # Initialize vocabulary and build vocab
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions.tolist())
        else:
            self.vocab = vocab

    def __len__(self):
        # return len(self.df)
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img_pil = Image.open(img_location).convert("RGB")

        # do some random augmentations
        if not self.do_augmentation:
            img = transforms(img_pil)
        else:
            img = transforms_advanced(img_pil)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self, pad_idx, batch_first=False, max_len=0):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self._max_len = max_len

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        if self._max_len > 0:
            if targets.shape[1] >= self._max_len:
                targets = targets[:, :self._max_len]
            else:
                pad_tensor = torch.ones(size=(targets.shape[0], self._max_len - targets.shape[1]),
                                        dtype=torch.long) * self.pad_idx
                targets = torch.cat([targets, pad_tensor], dim=1)

        return imgs, targets
