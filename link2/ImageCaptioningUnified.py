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
import os
import time
from collections import OrderedDict

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
        data_validation_images_path=f"{cwd}/data/flickr8k/Flickr8kValidationImages/",
        data_validation_captions=f"{cwd}/data/flickr8k/captions_validation.txt",
        data_test_images_path=f"{cwd}/data/flickr8k/Flickr8kTestImages/",
        data_test_captions=f"{cwd}/data/flickr8k/captions_test.txt",

        # Running parameters #
        freq_threshold=2,
        batch_size=256,
        data_limit=512,  # dbg
        max_val_show=10,  # tsb
        num_worker=4,
        learning_rate=1e-3,
        num_epochs=100,
        device_ids=[0, 1],
        print_every=5,
        save_every_epochs=10,

        # Common hyperparams #
        seq_len=30,
        embed_size=256,
        image_dimension=2048,

        # lstm parameters #
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,

        # Transformer parameters #
        num_decoder_layers=4,
        nhead=8,
        d_model=256,
        dim_feedforward=512,
        dropout=0.3,
):
    # Running unique ids
    id_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = SummaryWriter(log_dir=f"{cwd}/tensorboard/link4/{run_mode}_images_{id_run}")

    # Train dataset and dataloader
    dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions,
                                  transform=transforms, data_limit=data_limit, freq_threshold=freq_threshold)
    pad_idx = dataset_train.vocab.stoi["<PAD>"]
    sos_idx = dataset_train.vocab.stoi["<SOS>"]
    eos_idx = dataset_train.vocab.stoi["<EOS>"]
    idx2word = dataset_train.vocab.itos
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker, shuffle=True,
                                   collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                          max_len=seq_len))

    # Validation dataset and dataloader
    dataset_validation = FlickrDataset(root_dir=data_validation_images_path, captions_file=data_validation_captions,
                                       transform=transforms, vocab=dataset_train.vocab, data_limit=data_limit)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size, num_workers=num_worker,
                                        shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                                             max_len=seq_len))
    dataset_test = FlickrDataset(root_dir=data_test_images_path, captions_file=data_test_captions,
                                 transform=transforms, vocab=dataset_train.vocab, data_limit=data_limit)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=num_worker,
                                  shuffle=True, collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True,
                                                                       max_len=seq_len))
    vocab_size = len(dataset_train.vocab)

    # Device handling
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"device to run on {device}")

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
        number_parameters = model.num_parameters
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
        number_parameters = model.num_parameters
    else:
        raise Exception(f"no run_mode={run_mode}")
    print(f"total number parameters={number_parameters:,}")
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    # model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset_train.vocab.stoi["<PAD>"], reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    print(f"lr={scheduler.optimizer.param_groups[0]['lr']}")

    # 5) Training Job from above configs
    num_batches = len(data_loader_train)
    num_of_pics_to_show = 3

    time_start_training = time.time()
    for epoch in range(num_epochs):
        loss_train = 0
        num_samples_train = 0
        for idx, (image, captions) in enumerate(data_loader_train):
            image, captions = image.to(device), captions.to(device)
            captions = captions[:, 0:-1]  # drip
            # captions = captions[:, 1:]
            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward training
            if run_mode == "transformer":
                tgt_key_padding_mask = captions == pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
                tgt_key_padding_mask.to(device)
                outputs = model.module.forward(image, captions, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                outputs = outputs[:, :-1]
            else:
                outputs, attentions = model.module.forward(image, captions)

            # Calculate the batch loss.
            targets = captions[:, 1:]
            loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))

            # Backward pass.
            loss.backward()
            loss_train += loss.item()
            num_samples_train += image.shape[0]

            # Update the parameters in the optimizer.
            optimizer.step()

            if idx % print_every == 0:
                epoch_batch_info_str = f"E{epoch:03d}/{num_epochs:03d} i{idx:03d}/{num_batches:03d} loss={loss_train / num_samples_train:.5f}"
                print(f"{epoch_batch_info_str} device={device}")

                model.eval()
                # Transformer: sending images and captions to the tensorboard
                if run_mode == "transformer":
                    with torch.no_grad():
                        dataiter = iter(data_loader_validation)
                        img, _ = next(dataiter)
                        img = img.to(device)
                        img = img[:max_val_show, :, :]
                        captions_pred_batch = greedy_decoding(model, img, sos_idx, eos_idx, pad_idx, idx2word,
                                                              max_len=seq_len - 1, device=device, tgt_mask=tgt_mask)
                        for i, caption in enumerate(captions_pred_batch):
                            caption = ' '.join(caption)
                            show_image(img[i], title=epoch_batch_info_str + " T:" + caption, tb=tb)

                else:
                    with torch.no_grad():
                        for _ in range(num_of_pics_to_show):
                            dataiter = iter(data_loader_validation)
                            img, _ = next(dataiter)
                            features = model.module.encoder(
                                img[0:1].to(device))  # drip: added module for parallelization
                            caps, alphas = model.module.decoder.generate_caption(features,
                                                                                 vocab=dataset_train.vocab)  # drip: added module for parallelization
                            caption = ' '.join(caps)
                            show_image(img[0], title=epoch_batch_info_str + " L:" + caption, tb=tb)

                model.train()

        loss_train_mean = loss_train / num_samples_train
        # Compute validation loss
        loss_val = 0
        num_samples_val = 0
        for idx, (image, captions) in enumerate(data_loader_validation):
            image, captions = image.to(device), captions.to(device)
            captions = captions[:, 1:]
            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            if run_mode == "transformer":
                tgt_key_padding_mask = captions != pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
                tgt_key_padding_mask.to(device)
                outputs = model.module.forward(image, captions, tgt_mask=tgt_mask)
                outputs = outputs[:, :-1]
            else:
                outputs, attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:, 1:]
            loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))
            loss_val += loss.item()
            num_samples_val += image.shape[0]

        loss_val_mean = loss_val / num_samples_val
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
        if epoch % save_every_epochs == 0 or epoch == num_epochs - 1:
            model_save_filename = f"{cwd}/models/{run_mode}_image_caption_model_state_{id_run}_{epoch:03d}.pth"
            torch.save(model, model_save_filename)
            memory_size = os.stat(model_save_filename).st_size / (1024 * 1024)
            print(f"save model {model_save_filename} size is {memory_size:.2f}MB")

    loss_test = 0
    num_samples_test = 0
    for idx, (image, captions) in enumerate(data_loader_test):
        image, captions = image.to(device), captions.to(device)
        captions = captions[:, 1:]
        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        if run_mode == "transformer":
            tgt_key_padding_mask = captions != pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
            tgt_key_padding_mask.to(device)
            outputs = model.module.forward(image, captions, tgt_mask=tgt_mask)
            outputs = outputs[:, :-1]
        else:
            outputs, attentions = model(image, captions)

        # Calculate the batch loss.
        targets = captions[:, 1:]
        loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))
        loss_test += loss.item()
        num_samples_test += image.shape[0]
    loss_test_mean = loss_test / num_samples_test
    print(f"loss_test_mean={loss_test_mean})")

    total_run_time = time.time() - time_start_training
    # measures and return values
    result = OrderedDict()
    result["id"] = id_run
    result["number_parameters"] = number_parameters
    result["total_run_time"] = total_run_time
    result["loss_train_mean"] = loss_train_mean
    result["loss_val_mean"] = loss_val_mean
    result["loss_test_mean"] = loss_test_mean
    result["memory_size"] = memory_size
    return result


if __name__ == "__main__":
    res = single_run(
        run_mode="lstm",
        num_epochs=100,
        data_limit=None,
        batch_size=1024)
    print(f"result={res}")
