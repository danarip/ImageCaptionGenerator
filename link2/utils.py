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


def plot_attention(img, result, attention_plot):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7, 7)

        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


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

