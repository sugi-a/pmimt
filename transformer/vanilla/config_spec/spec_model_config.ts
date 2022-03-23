type Int = number;

type RelPos = {
        'use_rel_pos': true,
        'rel_pos_max_dist': Int,
        'rel_pos_unique_per_head': boolean
    } | {
        'use_rel_pos': false
        'rel_pos_max_dist': null,
        'rel_pos_unique_per_head': null
    };

export type Config = {
    'd_model': Int,
    'n_heads': Int,
    'maxlen': Int,
    'ff_size': Int,
    'dropout_rate': number,
    'n_enc_blocks': Int,
    'n_dec_blocks': Int,
    'use_pos_enc': boolean,
    'use_pos_emb': boolean,
    'share_enc_dec_embedding': boolean,
    'vocab_size': Int
} & RelPos;
