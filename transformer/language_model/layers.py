from logging import getLogger; logger = getLogger(__name__)

import tensorflow as tf
from tensorflow import keras, nest
import numpy as np

from ..vanilla.layers import Decoder

class DecoderLanguageModel(keras.layers.Layer):
    def __init__(
            self,
            vocab_size,
            d_model,
            n_heads,
            ff_size,
            dropout_rate,
            maxlen,
            use_pos_emb,
            use_pos_enc,
            use_rel_pos,
            n_blocks,
            rel_pos_max_dist=None,
            rel_pos_unique_per_head=None):
        super().__init__()
            
        NORM_EPS = 1e-6 # epsilon of layer norm

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.maxlen = maxlen
        self.ff_size = ff_size
        self.dropout_rate = dropout_rate
        self.n_blocks = n_blocks
        self.use_pos_enc = use_pos_enc
        self.use_pos_emb = use_pos_emb
        self.use_rel_pos = use_rel_pos
        self.rel_pos_max_dist = rel_pos_max_dist
        self.rel_pos_unique_per_head = rel_pos_unique_per_head

        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            ff_size=ff_size,
            dropout_rate=dropout_rate,
            maxlen=maxlen,
            use_pos_emb=use_pos_emb,
            use_pos_enc=use_pos_enc,
            use_rel_pos=use_rel_pos,
            n_blocks=n_blocks,
            norm_eps=NORM_EPS,
            in_emb=None,
            rel_pos_max_dist=rel_pos_max_dist,
            rel_pos_unique_per_head=rel_pos_unique_per_head,
            context=False,
            name='decoder')


    def call(self, x, training, offsets=None, cache=None):
        return self.decoder(
            x,
            self_attn_bias='causal_bias',
            training=training,
            cache=cache,
            offsets=offsets)
    

    def create_cache(self, batch_size):
        return self.decoder.create_cache(batch_size)
    

    def permute_cache(self, cache, permutation):
        self.decoder.permute_cache(cache, permutation)


    @classmethod
    def from_config(cls, config, **kwargs):
        """Create an instance with configuration defined in `transformer_net.ts`
        """
        c = config
        return cls(
            vocab_size=c['vocab_size'],
            d_model=c['d_model'],
            n_heads=c['n_heads'],
            ff_size=c['ff_size'],
            dropout_rate=c['dropout_rate'],
            maxlen=c['maxlen'],
            use_pos_enc=c['use_pos_enc'],
            use_pos_emb=c['use_pos_emb'],
            use_rel_pos=c['use_rel_pos'],
            n_blocks=c['n_blocks'],
            rel_pos_max_dist=c['rel_pos_max_dist'],
            rel_pos_unique_per_head=c['rel_pos_unique_per_head'],
            **kwargs)
