import matplotlib.pyplot as plt
import torch

from definitions import cwd


def show_image(img, title=None, tb=None):
    """Imshow for Tensor."""

    # unnormalize
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    if tb is not None:
        tb.add_image(title, img)
    else:
        img = img.numpy().transpose((1, 2, 0))
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


def save_model(model2save,
               num_epochs,
               embed_size,
               vocab_size,
               attention_dim,
               encoder_dim,
               decoder_dim,
               id_for_save):
    # helper function to save the model
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': embed_size,
        'vocab_size': vocab_size,
        'attention_dim': attention_dim,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'state_dict': model2save.state_dict()
    }
    torch.save(model_state, f"{cwd}/models/attention_model_state_{id_for_save}.pth")


def load_file(path):
    f = open(path, "r")
    s = f.read()
    f.close()
    return s


def get_caption_from_index(idx2word, caption_idx, pad_idx=0):
    caption_true = " ".join([idx2word[idx] for idx in caption_idx if idx != pad_idx])
    return caption_true


def captions_to_words(captions_indices,  # input is captions as indices
                      vocab):
    captions_words = list()
    for caption_index in captions_indices:
        captions_word = list()
        for idx in caption_index:
            captions_word.append(vocab.itos[idx])
            if vocab.eos_idx is not None and idx==vocab.eos_idx:
                break
        captions_words.append(captions_word)
    return captions_words


def words_to_sentences(words):
    return [" ".join(caption) for caption in words]