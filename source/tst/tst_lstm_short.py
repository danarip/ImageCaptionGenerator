from source.ImageCaptioning import single_run

if __name__ == "__main__":
    res = single_run(
        tb_run_name="tst_lstm_short",
        run_mode="lstm",
        freq_threshold=5,
        num_epochs=10,
        data_limit=None,
        batch_size=512,  # Suitable for 2 x A5000 GPUs
        seq_len=15,  # Sequence length (for padding)
        embed_size=256,  # Embedding of words
        # LSTM params
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=256,
        print_every=10,
        num_worker=4
    )
    print(f"result={res}")
