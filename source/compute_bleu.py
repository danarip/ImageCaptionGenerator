import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image

from definitions import cwd
from source.data_preprocessing import Vocabulary
from source.data_preprocessing import FlickrDataset, transforms, CapsCollate
from nltk.translate.bleu_score import sentence_bleu
from source.decoding_utils import greedy_decoding_transformer
from data_preprocessing import transforms

"""
This script goes over the test dataset and compute the BLEU score for the generated captions.
"""

def compute_bleu(
        data_train_images_path=f"{cwd}/data/flickr8k/Flickr8kTrainImages/",
        data_train_captions=f"{cwd}/data/flickr8k/captions_train.txt",
        data_test_images_path=f"{cwd}/data/flickr8k/Flickr8kTestImages/",
        data_test_captions=f"{cwd}/data/flickr8k/captions_test.txt",
        network_file="lstm_image_caption_model_state_20230123_000127_039.pth",
        seq_len=30,
        data_limit=None,
        freq_threshold=2,
        num_worker=4,
        batch_size=64
):
    dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions,
                                  transform=transforms, data_limit=data_limit, freq_threshold=freq_threshold)
    pad_idx = dataset_train.vocab.stoi["<PAD>"]
    sos_idx = dataset_train.vocab.stoi["<SOS>"]
    eos_idx = dataset_train.vocab.stoi["<EOS>"]
    idx2word = dataset_train.vocab.itos
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker, shuffle=False,
                                   collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                          max_len=seq_len))
    dataset_test = FlickrDataset(root_dir=data_test_images_path, captions_file=data_test_captions,
                                 transform=transforms, vocab=dataset_train.vocab, data_limit=data_limit)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=num_worker,
                                  shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                                       max_len=seq_len))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device to rune on {device}")
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len - 1).to(device) != 0

    df = pd.read_csv(data_test_captions)
    imgs = df["image"]
    captions = df["caption"]
    img2captions = dict()
    for i in range(500):
        if imgs[i] not in img2captions.keys():
            img2captions[imgs[i]] = list()
        img2captions[imgs[i]].append(Vocabulary.tokenize(captions[i]))

    full_path = f"{cwd}/models/{network_file}"
    model = torch.load(full_path).to(device)
    model.eval()
    bleu = list()
    for i, img in enumerate(imgs):
        img_location = f"{data_test_images_path}/{img}"
        image = Image.open(img_location).convert("RGB")
        image = transforms(image)
        image = image.type(torch.FloatTensor).unsqueeze(0).to(device)
        with torch.no_grad():
            if "lstm" in network_file:
                features = model.module.encoder(image.to(device))  # drip: added module for parallelization
                caption, _ = model.module.decoder.generate_captions_greedy_lstm(features,
                                                                                vocab=dataset_train.vocab)  # drip: added module for parallelization
            else:
                caption = greedy_decoding_transformer(model, image, sos_idx, eos_idx, pad_idx, idx2word,
                                                      max_len=seq_len - 1, device=device, tgt_mask=tgt_mask)
                caption = caption[0]

        score = sentence_bleu(img2captions[img], caption)
        bleu.append(score)
        if i % 10==0:
            print(f"i-{i} mean = {np.mean(bleu)}")


if __name__ == "__main__":
    compute_bleu()