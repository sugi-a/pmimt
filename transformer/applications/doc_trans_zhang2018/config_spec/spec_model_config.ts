type Int = number;

export type Config = {
    d_model: Int,
    n_heads: Int,
    maxlen: Int,
    ff_size: Int,
    dropout_rate: number,
    n_enc_blocks: Int,
    n_dec_blocks: Int,
    n_ctx_blocks: Int,
    use_pos_enc: boolean,
    use_pos_emb: boolean,
    share_enc_dec_embedding: boolean,
    vocab_size: Int
}
