import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from source.utils import show_image
from definitions import cwd
from source.data_preprocessing import FlickrDataset, CapsCollate, transforms


# ## 6 Visualizing the attentions
# Defining helper functions
# Given the image generate captions and attention scores. Plot the attention scores in the image


# generate caption
def get_caps_from(features_tensors, model, device, vocab):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0], title=caption)

    return caps, alphas


def plot_attention(img, result, attention_plot):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure()
    ax = fig.add_subplot(4, 4, 1)
    img = ax.imshow(img)

    len_result = len(result)
    for l in range(min(len_result, 15)):
        temp_att = attention_plot[l].reshape(7, 7)

        ax = fig.add_subplot(4, 4, l + 2)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    plt.show(block=True)


# Show attention
def show_attention_old(data_loader_validation, model, device, vocab, total_number):
    # show any 1
    for i in range(total_number):
        dataiter = iter(data_loader_validation)
        images, _ = next(dataiter)

        img = images[0].detach().clone()
        img1 = images[0].detach().clone()
        caps, alphas = get_caps_from(img.unsqueeze(0), model=model, device=device, vocab=vocab)

        plot_attention(img1, caps, alphas)
    input()


def show_attention():
    model_path = "/home/dotan/projects/ImageCaptionGenerator/models/attention_model_state_20230119_163205_040.pth"
    id_run = model_path.split("/")[-1]

    tb = SummaryWriter(log_dir= f"{cwd}/tensorboard/link2/{id_run}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device to rune on {device}")

    # Train
    data_train_images_path = f"{cwd}/data/flickr8k/Flickr8kTrainImages/"
    data_train_captions = f"{cwd}/data/flickr8k/captions_train.txt"
    data_validation_images_path = f"{cwd}/data/flickr8k/Flickr8kTrainImages/"
    data_validation_captions = f"{cwd}/data/flickr8k/captions_train.txt"

    # Train dataset for the vocab
    dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions,
                                  transform=transforms)
    pad_idx = dataset_train.vocab.stoi["<PAD>"]
    # Validation dataset for the images
    dataset_validation = FlickrDataset(root_dir=data_validation_images_path, captions_file=data_validation_captions,
                                       transform=transforms, vocab=dataset_train.vocab)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=1024, num_workers=1,
                                        shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True))

    kwargs = {"embed_size": 256, "vocab_size": len(dataset_train.vocab), "attention_dim": 256, "encoder_dim": 2048,
              "decoder_dim": 256, "device":device}

    model = torch.load(model_path)
    model.eval()
    show_attention_old(data_loader_validation, model, device, dataset_train.vocab, total_number=3)


if __name__ == "__main__":
    show_attention()
