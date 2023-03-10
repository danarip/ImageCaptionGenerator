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
import json
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from definitions import cwd
from source.data_preprocessing import FlickrDataset, CapsCollate
from source import utils
from source.networks_transformer import EncoderDecoderTransformer
from source.networks_lstm import EncoderDecoderLSTMAttention
from source.data_preprocessing import transforms
from source.decoding_utils import greedy_decoding_transformer


def compute_loss_on_dataset(model,
                            data_loader_validation,
                            device,
                            criterion,
                            run_mode,
                            vocab,
                            seq_len,
                            tgt_mask=None):
    loss_val = 0
    num_samples_val = 0
    model.eval()
    with torch.no_grad():
        for idx, (img, captions) in enumerate(data_loader_validation):
            img, captions = img.to(device), captions.to(device)
            # captions = captions[:, 1:]

            if run_mode == "transformer":
                captions_pred_idx, captions_pred_prob = greedy_decoding_transformer(model, img, vocab,
                                                                                    max_len=seq_len,
                                                                                    device=device, tgt_mask=tgt_mask)
            else:  # run_mode == "lstm"
                features = model.module.encoder(img)  # drip: added module for parallelization
                captions_pred_idx, alphas, captions_pred_prob = model.module.decoder.generate_captions_greedy_lstm(
                    features,
                    max_len=seq_len,
                    vocab=vocab)

                # Calculate the batch loss.
            targets = captions[:, 1:]
            loss = criterion(captions_pred_prob.contiguous().view(-1, vocab.size()), targets.reshape(-1))
            loss_val += loss.item()
            num_samples_val += img.shape[0]
    model.train()

    return loss_val, num_samples_val


def single_run(
        run_mode="lstm",  # Can be lstm or transformer
        tb_run_name="cleaning01",  # Tensorboard sub-folder to keep the results.

        # locations of the training / validation data #
        data_train_images_path=f"{cwd}/data/flickr8k/Flickr8kTrainImages/",  # Input: train Flickr images
        data_train_captions=f"{cwd}/data/flickr8k/captions_train.txt",  # Input: captions of Flickr train
        data_validation_images_path=f"{cwd}/data/flickr8k/Flickr8kValidationImages/",  # validation images
        data_validation_captions=f"{cwd}/data/flickr8k/captions_validation.txt",  # validation captions
        data_test_images_path=f"{cwd}/data/flickr8k/Flickr8kTestImages/",  # test images
        data_test_captions=f"{cwd}/data/flickr8k/captions_test.txt",  # test captions
        do_augmentation=False,  # wheter we would like to have images augmentations.

        # Running parameters #
        freq_threshold=2,  # minimal number of words in the vocabulary
        batch_size=256,  # batch size
        data_limit=512,  # debug: limit the dataset for quick run. None skip this limit.
        max_val_show=10,  #
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
    # local_memory = locals()
    # print(f"{json.dumps(local_memory, sort_keys=True, indent=4)}")
    # Running unique ids
    id_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = SummaryWriter(log_dir=f"{cwd}/tensorboard/{tb_run_name}/{run_mode}_images_{id_run}")

    # Train dataset and dataloader
    dataset_train = FlickrDataset(root_dir=data_train_images_path, captions_file=data_train_captions,
                                  transform=transforms, data_limit=data_limit, freq_threshold=freq_threshold,
                                  do_augmentation=do_augmentation)
    vocab = dataset_train.vocab
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker, shuffle=True,
                                   collate_fn=CapsCollate(vocab=vocab, batch_first=True,
                                                          max_len=seq_len))

    # Validation dataset and dataloader
    dataset_validation = FlickrDataset(root_dir=data_validation_images_path, captions_file=data_validation_captions,
                                       transform=transforms, vocab=dataset_train.vocab, data_limit=data_limit)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size, num_workers=num_worker,
                                        shuffle=True, collate_fn=CapsCollate(vocab=vocab, batch_first=True,
                                                                             max_len=seq_len))
    dataset_test = FlickrDataset(root_dir=data_test_images_path, captions_file=data_test_captions,
                                 transform=transforms, vocab=dataset_train.vocab, data_limit=data_limit)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=num_worker,
                                  shuffle=True, collate_fn=CapsCollate(vocab=vocab, batch_first=True,
                                                                       max_len=seq_len))
    vocab_size = len(dataset_train.vocab)
    max_val_show = min(max_val_show, batch_size)
    print(
        f"lengths(samples): dataset_train={len(dataset_train)}, dataset_validation={len(dataset_validation)}, dataset_test={len(dataset_test)}")
    print(
        f"lengths(batches): data_loader_train={len(data_loader_train)}, data_loader_validation={len(data_loader_validation)}, data_loader_test={len(data_loader_test)}, vocab_size={vocab_size}")

    # Choose 30 pictures to run with along the training of both

    # Device handling
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"device to run on {device}")

    tgt_mask = None if run_mode == "lstm" else nn.Transformer.generate_square_subsequent_mask(seq_len).to(
        device) != 0
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

    time_start_training = time.time()
    for epoch in range(num_epochs):
        time_start_epoch = time.time()
        loss_train = 0
        num_samples_train = 0
        for idx, (image, captions) in enumerate(data_loader_train):
            image, captions = image.to(device), captions.to(device)
            targets = captions[:, 1:]

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward training
            if run_mode == "transformer":
                tgt_key_padding_mask = captions == vocab.pad_idx  # tgt_key_padding_mask: (T) for unbatched input otherwise (N, T)
                tgt_key_padding_mask.to(device)
                outputs = model.module.forward(image, captions, tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask)
                outputs = outputs[:, 0:-1]
            else:
                outputs, attentions = model.module.forward(image, captions[:, 0:-1], seq_len)

            # Calculate the batch loss.
            loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.reshape(-1))

            # Backward pass.
            loss.backward()
            loss_train += loss.item()
            num_samples_train += image.shape[0]

            # Update the parameters in the optimizer.
            optimizer.step()

            if idx % print_every == 0:
                time_start_tb = time.time()
                epoch_batch_info_str = f"E{epoch:03d}/{num_epochs:03d} I{idx:03d}/{num_batches:03d} loss={loss_train / num_samples_train:.5f} device={device}"

                model.eval()
                # Transformer: sending images and captions to the tensorboard
                dataiter = iter(data_loader_validation)
                img, caption_true_idx = next(dataiter)
                img = img[:max_val_show, :, :].to(device)
                captions_true_words = utils.captions_to_words(caption_true_idx.tolist(), dataset_train.vocab)
                captions_true_sentences = utils.words_to_sentences(captions_true_words)
                with torch.no_grad():
                    if run_mode == "transformer":
                        captions_pred_batch, y_pred_batch = greedy_decoding_transformer(model, img, vocab,
                                                                                        max_len=seq_len,
                                                                                        device=device,
                                                                                        tgt_mask=tgt_mask)
                        for i, caption in enumerate(captions_pred_batch):
                            caption = ' '.join(caption)
                            title = f"{epoch_batch_info_str} T{d_model},{dim_feedforward}|{caption}|{captions_true_sentences[i]}"
                            utils.show_image(img[i], title=title, tb=tb)

                    else:  # run_mode == "lstm"
                        features = model.module.encoder(img)  # drip: added module for parallelization
                        captions_pred_idx, alphas, captions_pred_prob = model.module.decoder.generate_captions_greedy_lstm(
                            features,
                            max_len=seq_len,
                            vocab=dataset_train.vocab)  # drip: added module for parallelization
                        captions_pred_idx = captions_pred_idx.tolist()
                        captions_pred_words = utils.captions_to_words(captions_pred_idx, vocab=vocab)
                        caption_sentences = utils.words_to_sentences(captions_pred_words)
                        for i in range(max_val_show):
                            title = f"{epoch_batch_info_str} L{decoder_dim}|{caption_sentences[i]}|{captions_true_sentences[i]}"
                            utils.show_image(img[i], title=title, tb=tb)
                print(title)  # show a single title
                model.train()
                print(f"tb time={time.time() - time_start_tb}")
        print(f"epoch_time={time.time() - time_start_epoch}")

        loss_train_mean = loss_train / num_samples_train
        # Compute validation loss
        loss_val, num_samples_val = compute_loss_on_dataset(model,
                                                            data_loader_validation,
                                                            device,
                                                            criterion,
                                                            run_mode,
                                                            vocab,
                                                            seq_len,
                                                            tgt_mask=tgt_mask)

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

    loss_test, num_samples_test = compute_loss_on_dataset(model,
                                                          data_loader_test,
                                                          device,
                                                          criterion,
                                                          run_mode,
                                                          vocab,
                                                          seq_len,
                                                          tgt_mask=tgt_mask)
    loss_test_mean = loss_test / num_samples_test
    print(f"loss_train_mean={loss_train_mean})")
    print(f"loss_test_mean={loss_test_mean})")

    total_run_time = time.time() - time_start_training
    # measures and return values
    result = OrderedDict()
    result["run_mode"] = run_mode
    result["id"] = id_run
    result["number_parameters"] = number_parameters
    result["total_run_time"] = total_run_time
    result["loss_train_mean"] = loss_train_mean
    result["loss_val_mean"] = loss_val_mean
    result["loss_test_mean"] = loss_test_mean
    result["memory_size"] = memory_size
    result["total_run_time"] = total_run_time
    # result["locals"] = local_memory
    return result


if __name__ == "__main__":
    res = single_run(
        run_mode="transformer",
        num_epochs=10,
        data_limit=None,
        batch_size=512)
    print(f"result={res}")
