from logging import getLogger; logger = getLogger(__name__)

import tensorflow as tf
from tensorflow import keras, nest
import numpy as np

from ...vanilla import layers as vl
from ...utils.beam_search import beam_search


class NormResidualWrapperWithContextGate(keras.layers.Layer):
    def __init__(
        self, layer_constructor, dropout_rate, norm_eps, pretrain, **kwargs):
        """
        Wraps the given layer (usually MH-attention or feed-foward)
        with layer-norm, dropout, and residual connection.
        Args:
            layer_constructor:
                A callable which creates the sublayer instance. Instanciating
                the sublayer in this __init__ is important because it enables
                this class to track the weights of the sublayer.
        """
        super().__init__(**kwargs, trainable=not pretrain)

        self.pretrain = pretrain

        self.layer_norm = keras.layers.LayerNormalization(epsilon=norm_eps)
        self.layer = layer_constructor()
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.wi = keras.layers.Dense(1, use_bias=False, kernel_initializer='zeros')
        self.ws = keras.layers.Dense(1, use_bias=False, kernel_initializer='zeros')


    def call(self, x, *args, training, **kwargs):
        if self.pretrain:
            return x
        else:
            y = self.layer_norm(x)
            y = self.layer(y, *args, training=training, **kwargs)
            y = self.dropout(y, training=training)
            l = tf.math.sigmoid(self.wi(x) + self.ws(y))
            return l * x + (1 - l) * y


class MainEncoderBlock(keras.layers.Layer):
    def __init__(
            self, d_model, n_heads, dropout_rate, ff_size, norm_eps, pretrain, **kwargs):
        super().__init__(**kwargs)

        self.self_attn = vl.NormResidualWrapper(
                lambda: vl.SelfAttention( d_model, n_heads, dropout_rate),
                dropout_rate=dropout_rate,
                norm_eps=norm_eps,
                trainable=pretrain)
        
        self.ctx_attn = NormResidualWrapperWithContextGate(
            lambda: vl.MultiheadAttention( d_model, n_heads, dropout_rate),
            dropout_rate=dropout_rate,
            norm_eps=norm_eps,
            pretrain=pretrain)

        self.ff = vl.NormResidualWrapper(
            lambda: vl.Feedforward(ff_size, dropout_rate),
            dropout_rate,
            norm_eps,
            trainable=pretrain)


    def call(self, x, self_attn_bias, ctx, ctx_attn_bias, training):
        y = self.self_attn(x, self_attn_bias, training=training)
        y = self.ctx_attn(y, ctx, ctx_attn_bias, training=training)
        y = self.ff(y, training=training)
        return y


class MainEncoder(keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        ff_size,
        dropout_rate,
        norm_eps,
        maxlen, 
        n_blocks,
        use_pos_enc,
        use_pos_emb,
        pretrain,
        in_emb=None,
        **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        if in_emb is None:
            self.in_emb = vl.EmbeddingLayer(
                vocab_size, d_model, scale=True, trainable=pretrain)
        else:
            self.in_emb = in_emb
        if use_pos_enc: self.pos_enc = vl.positional_encoding(maxlen, d_model)
        if use_pos_emb:
            self.pos_emb = self.add_weight(
                name='pos_emb',
                shape=[maxlen, d_model],
                dtype=tf.float32,
                trainable=pretrain)
        self.post_emb_dropout = keras.layers.Dropout(dropout_rate)

        self.blocks = [
            MainEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                ff_size=ff_size,
                norm_eps=norm_eps, 
                pretrain=pretrain,
                name=f'layer_{i}')
            for i in range(n_blocks)
        ]

        self.output_norm = keras.layers.LayerNormalization(
            epsilon=norm_eps, trainable=pretrain)


    def call(self, x, self_attn_bias, ctx, ctx_attn_bias, training):
        """
        Args:
            x: <[B, L], int32>. value in [0, V - 1].
            self_attn_bias: <[(1|B), 1, (1|L), (1|L)], float32>
                if 'pading_bias' is specified, padding bias tensor is used
        """
        # Embedding [batch, length, emb_size]
        y = self.in_emb(x)

        if hasattr(self, 'pos_enc'):
            y += self.pos_enc[:tf.shape(y)[1]]

        if hasattr(self, 'pos_emb'):
            y += self.pos_emb[:tf.shape(y)[1]]

        y = self.post_emb_dropout(y, training=training)

        for block in self.blocks:
            y = block(
                y, self_attn_bias,
                ctx, ctx_attn_bias,
                training=training)

        return self.output_norm(y)


class DecoderBlock(keras.layers.Layer):
    def __init__(
            self, d_model, n_heads, dropout_rate, ff_size, norm_eps, pretrain, **kwargs):
        super().__init__(**kwargs)

        self.self_attn = vl.NormResidualWrapper(
                lambda: vl.SelfAttention(d_model, n_heads, dropout_rate),
                dropout_rate=dropout_rate,
                norm_eps=norm_eps,
                trainable=pretrain)
        
        self.enc_attn = vl.NormResidualWrapper(
            lambda: vl.MultiheadAttention(d_model, n_heads, dropout_rate),
            dropout_rate=dropout_rate,
            norm_eps=norm_eps,
            trainable=pretrain)

        self.ctx_attn = NormResidualWrapperWithContextGate(
            lambda: vl.MultiheadAttention(d_model, n_heads, dropout_rate),
            dropout_rate=dropout_rate,
            norm_eps=norm_eps,
            pretrain=pretrain)

        self.ff = vl.NormResidualWrapper(
            lambda: vl.Feedforward(ff_size, dropout_rate),
            dropout_rate,
            norm_eps,
            trainable=pretrain)


    def call(
            self,
            x, self_attn_bias,
            ctx, ctx_attn_bias,
            enc, enc_attn_bias,
            training,
            cache=None):
        y = self.self_attn(x, self_attn_bias, training=training, cache=cache)
        y = self.ctx_attn(y, ctx, ctx_attn_bias, training=training)
        y = self.enc_attn(y, enc, enc_attn_bias, training=training)
        y = self.ff(y, training=training)
        return y
    

    def create_cache(self, batch_size):
        return self.self_attn.layer.create_cache(batch_size)
    

    def permute_cache(self, cache, ids):
        self.self_attn.layer.permute_cache(cache, ids)


class Decoder(keras.layers.Layer):
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
            n_blocks,
            norm_eps,
            pretrain,
            in_emb=None,
            **kwargs):
        super().__init__(**kwargs)

        if in_emb is not None:
            self.in_emb = in_emb
        else:
            self.in_emb = vl.EmbeddingLayer(
                vocab_size, d_model, scale=True, trainable=pretrain)

        if use_pos_enc: self.pos_enc = vl.positional_encoding(maxlen, d_model)
        if use_pos_emb: self.pos_emb = self.add_weight(
            name='pos_emb', shape=[maxlen, d_model], dtype=tf.float32)

        self.post_emb_dropout = keras.layers.Dropout(dropout_rate)

        self.blocks = [
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
                ff_size=ff_size,
                norm_eps=norm_eps, 
                pretrain=pretrain,
                name=f'layer_{i}')
            for i in range(n_blocks)
        ]

        self.output_norm = keras.layers.LayerNormalization(
            epsilon=norm_eps, trainable=pretrain)


    def call(
            self,
            x, self_attn_bias,
            ctx, ctx_attn_bias,
            enc, enc_attn_bias,
            training,
            cache=None):
        """
        Notation:
            B := batch size
            L := length of the longest sequence in minibach (= shape(x)[1])
            ML := maximum length (constant defined in config)
            [l, r) := the range of the current input (0-index)
        Args:
            self_attn_bias: [(1 | B), 1, (1|r - l), (1|r)] | 'causal_bias'
                if 'causal_bias' is specified, causal bias is used.
            ctx_attn_bias: [(1|B), 1, (1|r-l), (1|len_ctx)]
            cache: {
                [`layer_${i}`]: {
                    'v': <[B, L_cache, E]>,
                    'k': <[B, L_cache, E]>},
                cur_pos: Tensor<[], int>
            } | None
        """
        # Prediction span: [l, r)
        l = 0 if cache is None else cache['cur_pos']
        r = l + tf.shape(x)[1]

        if cache is not None:
            cache['cur_pos'] = r

        # Generic biases
        if self_attn_bias == 'causal_bias':
            # [1, 1, r, r] -> [1, 1, r-l, r]
            self_attn_bias = vl.create_causal_bias(r)[:, :, l:]

        # Adjust the position-related tensors
        pos_enc = self.pos_enc[l: r] if hasattr(self, 'pos_enc') else None
        pos_emb = self.pos_emb[l: r] if hasattr(self, 'pos_emb') else None

        # Start of graph construction
        y = self.in_emb(x)
        if pos_enc is not None:
            y += pos_enc
        if pos_emb is not None:
            y += pos_emb
        
        y = self.post_emb_dropout(y, training=training)

        for i, block in enumerate(self.blocks):
            y = block(
                y,
                self_attn_bias=self_attn_bias,
                ctx=ctx,
                ctx_attn_bias=ctx_attn_bias,
                enc=enc,
                enc_attn_bias=enc_attn_bias,
                training=training,
                cache=cache[f'layer_{i}'] if cache is not None else None
            )
        
        y = self.output_norm(y)

        return self.in_emb.emb2logits(y)
    

    def create_cache(self, batch_size):
        cache = {'cur_pos': tf.constant(0)}
        shape = {'cur_pos': tf.TensorShape([])}
        for i, block in enumerate(self.blocks):
            key = f'layer_{i}'
            cache[key], shape[key] = block.create_cache(batch_size)
        return cache, shape


    def permute_cache(self, cache, ids):
        """Rearrange each of the cached Tensors along the batch dim."""
        for i, block in enumerate(self.blocks):
            block.permute_cache(cache[f'layer_{i}'], ids)


class DocTransformer(keras.layers.Layer):
    def __init__(
            self,
            vocab_size,
            d_model,
            n_heads,
            maxlen,
            ff_size,
            dropout_rate,
            n_enc_blocks,
            n_dec_blocks,
            n_ctx_blocks,
            use_pos_enc,
            use_pos_emb,
            share_enc_dec_embedding,
            pretrain):
        super().__init__()

        NORM_EPS = 1e-6 # epsilon of layer norm

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.maxlen = maxlen
        self.ff_size = ff_size
        self.dropout_rate = dropout_rate
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.n_ctx_blocks = n_ctx_blocks
        self.use_pos_enc = use_pos_enc
        self.use_pos_emb = use_pos_emb
        self.share_enc_dec_embedding = share_enc_dec_embedding

        self.src_emb = vl.EmbeddingLayer(
            vocab_size, d_model, scale=True, trainable=pretrain)

        self.main_encoder = MainEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            ff_size=ff_size,
            dropout_rate=dropout_rate,
            norm_eps=NORM_EPS,
            maxlen=maxlen, 
            n_blocks=n_enc_blocks,
            use_pos_enc=use_pos_enc,
            use_pos_emb=use_pos_emb,
            pretrain=pretrain,
            in_emb=self.src_emb,
            name='main_encoder')

        self.ctx_encoder = vl.Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            ff_size=ff_size,
            dropout_rate=dropout_rate,
            norm_eps=NORM_EPS,
            maxlen=maxlen, 
            n_blocks=n_ctx_blocks,
            use_pos_enc=use_pos_enc,
            use_pos_emb=use_pos_emb,
            use_rel_pos=False,
            rel_pos_max_dist=None,
            rel_pos_unique_per_head=None,
            name='ctx_encoder',
            trainable=not pretrain,
            in_emb=self.src_emb)

        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            ff_size=ff_size,
            dropout_rate=dropout_rate,
            maxlen=maxlen,
            use_pos_emb=use_pos_emb,
            use_pos_enc=use_pos_enc,
            n_blocks=n_dec_blocks,
            norm_eps=NORM_EPS,
            pretrain=pretrain,
            in_emb=self.ctx_encoder.in_emb if share_enc_dec_embedding else None,
            name='decoder')
    

    def call(self, ctx_input, enc_input, dec_input, training):
        """
        Args:
            enc_input: [B, L_enc], int32
            dec_input: [B, L_dec], int32
            training: bool
            ret_embedding: bool=False
        Returns:
            Tensor<[B, L_dec, E], float32> if ret_embedding.
            Tensor<[B, L_dec, V], int32> if not ret_embedding.
        """
        ctx_pad_bias = vl.seq_to_padding_bias(ctx_input)
        ctx_out = self.ctx_encoder(
            ctx_input, self_attn_bias=ctx_pad_bias, training=training)

        enc_pad_bias = vl.seq_to_padding_bias(enc_input)
        enc_out = self.main_encoder(
            enc_input,
            self_attn_bias=enc_pad_bias,
            ctx=ctx_out,
            ctx_attn_bias=ctx_pad_bias,
            training=training)

        dec_out = self.decoder(
            dec_input,
            self_attn_bias='causal_bias',
            ctx=ctx_out,
            ctx_attn_bias=ctx_pad_bias,
            enc=enc_out,
            enc_attn_bias=enc_pad_bias,
            training=training,
            cache=None)
        
        return dec_out


    def beam_search_decode(
            self, c, x, sos, eos, beam_size, maxlen, length_penalty_fn=None):
        
        ctx_pad_bias = vl.seq_to_padding_bias(c)
        ctx_out = self.ctx_encoder(
            c, self_attn_bias=ctx_pad_bias, training=False)

        enc_pad_bias = vl.seq_to_padding_bias(x)
        enc_out = self.main_encoder(
            x,
            self_attn_bias=enc_pad_bias,
            ctx=ctx_out,
            ctx_attn_bias=ctx_pad_bias,
            training=False)
        B, L, E = (tf.shape(enc_out)[i] for i in range(3))
        K = beam_size

        sos = tf.broadcast_to(sos, [B])

        maxlen = self.maxlen if maxlen is None \
            else tf.math.minimum(self.maxlen, maxlen)

        rep_ctx_out, rep_ctx_pad_bias, rep_enc_out, rep_enc_pad_bias \
            = [tf.repeat(x, K, axis=0) for x in
                [ctx_out, ctx_pad_bias, enc_out, enc_pad_bias]]

        cache, shape_inv = self.decoder.create_cache(B * K)

        def get_logits_fn(x):
            logits = self.decoder(
                x,
                self_attn_bias='causal_bias',
                ctx=rep_ctx_out,
                ctx_attn_bias=rep_ctx_pad_bias,
                enc=rep_enc_out,
                enc_attn_bias=rep_enc_pad_bias,
                training=False,
                cache=cache)
            return tf.nn.log_softmax(logits)


        def perm_batch_fn(batch_permutation):
            self.decoder.permute_cache(cache, batch_permutation)

        def get_state_fn():
            return cache

        def put_controlled_state_fn(cache_):
            nonlocal cache
            cache = cache_

        paths, scores = beam_search(
            get_logits_fn=get_logits_fn,
            perm_batch_fn=perm_batch_fn,
            sos=sos,
            eos=eos,
            beam_size=beam_size,
            maxlen=maxlen,
            pad=0,
            get_state_fn=get_state_fn,
            put_controlled_state_fn=put_controlled_state_fn,
            shape_invariants=shape_inv,
            length_penalty_fn=length_penalty_fn)

        return paths, scores


    @classmethod
    def from_config(cls, config, pretrain, **kwargs):
        c = config
        return cls(
            vocab_size=c['vocab_size'],
            d_model=c['d_model'],
            n_heads=c['n_heads'],
            maxlen=c['maxlen'],
            ff_size=c['ff_size'],
            dropout_rate=c['dropout_rate'],
            n_enc_blocks=c['n_enc_blocks'],
            n_dec_blocks=c['n_dec_blocks'],
            n_ctx_blocks=c['n_ctx_blocks'],
            use_pos_enc=c['use_pos_enc'],
            use_pos_emb=c['use_pos_emb'],
            share_enc_dec_embedding=c['share_enc_dec_embedding'],
            pretrain=pretrain,
            **kwargs)
