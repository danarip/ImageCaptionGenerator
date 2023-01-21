import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from definitions import cwd
from link2.data_preprocessing import FlickrDataset, transforms, CapsCollate
from link2.utils import load_file, show_image
from link2.decoding_utils import greedy_decoding


def single_run(
        data_train_images_path=f"{cwd}/data/flickr8k/Flickr8kTrainImages/",
        data_train_captions=f"{cwd}/data/flickr8k/captions_train.txt",
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
        batch_size=64,
        max_imgs=50
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

    captions = load_file(data_train_captions)
    captions = captions.split("\n")[1:max_imgs+1]
    filenames = [caption.split(",")[0] for caption in captions]
    print(captions)

    tb = SummaryWriter(log_dir=f"{cwd}/tensorboard/caption_dynamics/exp1/")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device to rune on {device}")
    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len - 1).to(device) != 0

    for network_file in network_files:
        full_path = f"{cwd}/models/{network_file}"
        model = torch.load(full_path).to(device)
        model.eval()
        with torch.no_grad():
            dataiter = iter(data_loader_train)
            imgs, _ = next(dataiter)
            imgs = imgs.type(torch.FloatTensor).to(device)
            for i in range(max_imgs):
                print(f"i={i}")
                if "lstm" in network_file:
                    features = model.module.encoder(imgs[i:i+1].to(device))  # drip: added module for parallelization
                    caps, alphas = model.module.decoder.generate_caption(features,
                                                                         vocab=dataset_train.vocab)  # drip: added module for parallelization
                    caption = ' '.join(caps)
                else:
                    captions_pred_batch = greedy_decoding(model, imgs[i:i+1], sos_idx, eos_idx, pad_idx, idx2word,
                                                          max_len=seq_len - 1, device=device, tgt_mask=tgt_mask)
                    caption = ' '.join(captions_pred_batch[0])
                show_image(imgs[i], title=f"{i} {network_file} {caption}", tb=tb)


if __name__ == "__main__":
    single_run()



