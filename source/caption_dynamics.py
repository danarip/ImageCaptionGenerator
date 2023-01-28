import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from definitions import cwd
from source.data_preprocessing import FlickrDataset, transforms, CapsCollate
from source.utils import load_file, show_image
from source.decoding_utils import greedy_decoding_transformer


def caption_dynamics(
        data_train_images_path=f"{cwd}/data/flickr8k/Flickr8kTrainImages/",
        data_train_captions=f"{cwd}/data/flickr8k/captions_train.txt",
        data_test_images_path=f"{cwd}/data/flickr8k/Flickr8kTestImages/",
        data_test_captions=f"{cwd}/data/flickr8k/captions_test.txt",
        path_to_network="models",
        network_files=["lstm_image_caption_model_state_20230121_215058_000.pth",
                       "lstm_image_caption_model_state_20230121_215058_010.pth",
                       "lstm_image_caption_model_state_20230121_215058_020.pth",
                       "lstm_image_caption_model_state_20230121_215058_030.pth",
                       "lstm_image_caption_model_state_20230121_215058_040.pth",
                       "lstm_image_caption_model_state_20230121_215058_050.pth",
                       "lstm_image_caption_model_state_20230121_215058_060.pth",
                       "lstm_image_caption_model_state_20230121_215058_070.pth",
                       "lstm_image_caption_model_state_20230121_215058_080.pth",
                       "transformer_image_caption_model_state_20230121_231314_000.pth",
                       "transformer_image_caption_model_state_20230121_231314_010.pth",
                       "transformer_image_caption_model_state_20230121_231314_020.pth",
                       "transformer_image_caption_model_state_20230121_231314_030.pth",
                       "transformer_image_caption_model_state_20230121_231314_040.pth",
                       "transformer_image_caption_model_state_20230121_231314_050.pth",
                       "transformer_image_caption_model_state_20230121_231314_060.pth",
                       "transformer_image_caption_model_state_20230121_231314_070.pth",
                       "transformer_image_caption_model_state_20230121_231314_080.pth"
                       ],
        seq_len=30,
        data_limit=None,
        freq_threshold=2,
        num_worker=4,
        batch_size=640,
        max_imgs=1
):
    dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions,
                                  transform=transforms, data_limit=data_limit, freq_threshold=freq_threshold, do_augmentation=False)
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



    captions = load_file(data_train_captions)
    captions = captions.split("\n")[1:max_imgs+1]
    filenames = [caption.split(",")[0] for caption in captions]
    print(captions)

    tb = SummaryWriter(log_dir=f"{cwd}/tensorboard/caption_dynamics/exp2/")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device to rune on {device}")
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len - 1).to(device) != 0

    dataiter = iter(data_loader_train)
    imgs, _ = next(dataiter)
    imgs = imgs.type(torch.FloatTensor).to(device)
    for i in range(0, max_imgs*5, 5):
        print(f"i={i}")
        for network_file in network_files:
            full_path = f"{cwd}/models/{network_file}"
            model = torch.load(full_path).to(device)
            model.eval()
            with torch.no_grad():
                if "lstm" in network_file:
                    features = model.module.encoder(imgs[i:i+1].to(device))  # drip: added module for parallelization
                    caps, alphas = model.module.decoder.generate_captions_greedy_lstm(features,
                                                                                      vocab=dataset_train.vocab)  # drip: added module for parallelization
                    caption = ' '.join(caps)
                    print(caption)
                else:
                    captions_pred_batch = greedy_decoding_transformer(model, imgs[i:i + 1], sos_idx, eos_idx, pad_idx, idx2word,
                                                                      max_len=seq_len - 1, device=device, tgt_mask=tgt_mask)
                    caption = ' '.join(captions_pred_batch[0])
                    print(caption)
                show_image(imgs[i], title=f"{i} {network_file} {caption}", tb=tb)


if __name__ == "__main__":
    caption_dynamics()



