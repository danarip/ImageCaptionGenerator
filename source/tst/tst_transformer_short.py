from source.ImageCaptioning import single_run

if __name__ == "__main__":
    res = single_run(
        tb_run_name="tst_transformer_short",
        run_mode="transformer",
        freq_threshold=5,
        num_epochs=1,
        data_limit=128,
        batch_size=128,  # Suitable for 2 x A5000 GPUs
        seq_len=15,  # Sequence length (for padding)
        embed_size=256,  # Embedding of words
        # Transformer params
        num_decoder_layers=4,
        nhead=8,
        d_model=200,
        dim_feedforward=512,
        dropout=0.2,

    )
    print(f"result={res}")
