import image2features


def tst_build_vocab():
    root_dir = "C:/deep_learning_046211/ImageCaptionGenerator"

    # build train set
    train_vocab = image2features.build_vocab(root_dir + "/Flickr8k_text/Flickr_8k.trainImages.txt",
                                             root_dir + "/Flickr8k_text/Flickr8k.token.txt")
    itos = train_vocab.vocab.get_itos()
    for i in range(30):
        print(f'{itos[i]}')


if __name__ == "__main__":
    tst_build_vocab()
