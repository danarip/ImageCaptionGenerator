'''
Taken from: https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k/notebook
and from: https://www.kaggle.com/code/mdteach/image-captioning-with-attention-pytorch

Attention type: Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015)
By: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio: https://arxiv.org/abs/1409.0473
Blog to explain the attention mechanism: https://machinelearningmastery.com/the-bahdanau-attention-mechanism

CNN-SamLynnEvans Architecture
https://github.com/senadkurtisi/pytorch-image-captioning
'''
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from root import cwd
from link2.data_preprocessing import FlickrDataset, CapsCollate
from link2.utils import show_image
from link2.networks_transformer import EncoderDecoderTransformer
from link2.networks_lstm import EncoderDecoderLSTMAttention
from link2.data_preprocessing import transforms
from link2.decoding_utils import greedy_decoding


def single_run(
        run_mode="lstm",

        # locations of the training / validation data #
        data_train_images_path=f"{cwd}/data/flickr8k/Flickr8kTrainImages/",
        data_train_captions=f"{cwd}/data/flickr8k/captions_train.txt",
        data_validation_images_path=f"{cwd}/data/flickr8k/Flickr8kTrainImages/",
        data_validation_captions=f"{cwd}/data/flickr8k/captions_train.txt",

        # Running parameters #
        batch_size=256,
        data_limit=512,
        max_val_show=10,
        num_worker=4,
        learning_rate=1e-3,
        num_epochs=100,
        cuda_number=1,
        print_every=5,
        save_every_epochs=10,

        # Common hyperparams #
        seq_len=30,
        embed_size=256,
        image_dimension=2048,

        # lstm parameters #
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=256,

        # Transformer parameters #
        num_decoder_layers=4,
        nhead=8,
        d_model=256,
        dim_feedforward=512,
        dropout=0.3,
):
    # Running unique ids
    id_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = SummaryWriter(log_dir=f"{cwd}/tensorboard/link3/{run_mode}_images_{id_run}")

    # Train dataset and dataloader
    dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions,
                                  transform=transforms, data_limit=data_limit)
    pad_idx = dataset_train.vocab.stoi["<PAD>"]
    sos_idx = dataset_train.vocab.stoi["<SOS>"]
    eos_idx = dataset_train.vocab.stoi["<EOS>"]
    idx2word = dataset_train.vocab.itos
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker, shuffle=True,
                                   collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                          max_len=seq_len))

    # Validation dataset and dataloader
    dataset_validation = FlickrDataset(root_dir=data_validation_images_path, captions_file=data_validation_captions,
                                       transform=transforms, vocab=dataset_train.vocab,data_limit=data_limit)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size, num_workers=num_worker,
                                        shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                                             max_len=seq_len))
    vocab_size = len(dataset_train.vocab)

    # Device handling
    device = torch.device(f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu")
    print(f"device to rune on {device}")

    # Defining the Model Architecture
    if run_mode == "transformer":
        model = EncoderDecoderTransformer(
            image_dimension=image_dimension,
            embed_size=embed_size,
            vocab_size=vocab_size,
            seq_len=seq_len,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,  # TransformerDecoderLayer: the number of expected features in the input
            nhead=nhead,  # TransformerDecoderLayer: the number of heads in the multiheadattention models
            dim_feedforward=dim_feedforward,
            device=device,
            dropout=dropout
        )
        # Explanation: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        # Square TxT mask for decoder, so the transformer won't cheat during the training and testing.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len - 1).to(device) != 0
    elif run_mode == "lstm":
        model = EncoderDecoderLSTMAttention(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            device=device)
    else:
        raise Exception(f"no run_mode={run_mode}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset_train.vocab.stoi["<PAD>"], reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    print(f"lr={scheduler.optimizer.param_groups[0]['lr']}")

    # 5) Training Job from above configs
    num_batches = len(data_loader_train)
    num_of_pics_to_show = 3

    time_start_training = time.time()
    for epoch in range(num_epochs):
        # time_start_epoch = time.time()
        loss_train = 0
        no_samples = 0
        for idx, (image, captions) in enumerate(data_loader_train):
            image, captions = image.to(device), captions.to(device)
            captions = captions[:, 1:]
            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            if run_mode == "transformer":
                tgt_key_padding_mask = captions != pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
                tgt_key_padding_mask.to(device)
                outputs = model(image, captions, tgt_mask=tgt_mask)
                outputs = outputs[:, :-1]
            else:
                outputs, attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:, 1:]
            loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))

            # Backward pass.
            loss.backward()
            loss_train += loss.item()
            no_samples += image.shape[0]

            # Update the parameters in the optimizer.
            optimizer.step()

            if idx % print_every == 0:
                epoch_batch_info_str = f"E{epoch:03d}/{num_epochs:03d} i{idx:03d}/{num_batches:03d} loss={loss_train / no_samples:.5f}"
                print(f"{epoch_batch_info_str} device={device}")

                # Transformer: sending images and captions to the tensorboard
                if run_mode=="transformer":
                    model.eval()
                    with torch.no_grad():
                        dataiter = iter(data_loader_validation)
                        img, _ = next(dataiter)
                        img = img.to(device)
                        img = img[:max_val_show, :, :]
                        captions_pred_batch = greedy_decoding(model, img, sos_idx, eos_idx, pad_idx, idx2word,
                                                              max_len=seq_len - 1, device=device, tgt_mask=tgt_mask)
                        for i, caption in enumerate(captions_pred_batch):
                            caption = ' '.join(caption)
                            show_image(img[i], title=epoch_batch_info_str + ":" + caption, tb=tb)

                    model.train()
        loss_train_mean = loss_train / no_samples
        # Compute validation loss
        loss_val = 0
        no_samples_val = 0
        for idx, (image, captions) in enumerate(data_loader_validation):
            image, captions = image.to(device), captions.to(device)
            captions = captions[:, 1:]
            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            if run_mode == "transformer":
                tgt_key_padding_mask = captions != pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
                tgt_key_padding_mask.to(device)
                outputs = model(image, captions, tgt_mask=tgt_mask)
                outputs = outputs[:, :-1]
            else:
                outputs, attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:, 1:]
            loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))
            loss_val += loss.item()
            no_samples_val += image.shape[0]

        loss_val_mean = loss_val / no_samples_val
        print(f"loss_train_mean={loss_train_mean})")
        print(f"loss_val_mean={loss_val_mean})")
        scheduler.step(loss_val)
        tb.add_scalars("loss", {"train": loss_train_mean, "val": loss_val_mean}, epoch)
        tb.add_scalar("learning_rate", scheduler.optimizer.param_groups[0]['lr'], epoch)
        print(f"lr={scheduler.optimizer.param_groups[0]['lr']}")
        time_from_training_start = (time.time() - time_start_training)
        expected_time = num_epochs / (epoch + 1) * time_from_training_start
        print(f"epoch time {time_from_training_start / (epoch + 1):.2f} [sec], elapsed_time {time_from_training_start}")
        print(f"expected total time {expected_time:.2f}")

        # save the latest model
        # save the latest model
        if epoch % save_every_epochs == 0 or epoch == num_epochs - 1:
            torch.save(model, f"{cwd}/models/transformer_image_caption_model_state_{id_run}_{epoch:03d}.pth")


if __name__ == "__main__":
    single_run()
