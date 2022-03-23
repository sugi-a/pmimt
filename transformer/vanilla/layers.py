from logging import getLogger; logger = getLogger(__name__)
import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest

from ..utils.beam_search import beam_search, sample_one
from .relative_position import RelativePositionMultiheadSelfAttention

"""
- Frequently used notation
    - Tensor<shape, dtype>
        - Specification of tf.Tensor with the shape and dtype
        - "Tensor" and "<>" may be omitted
        - e.g. Tensor<[2, 3], tf.int32> (integer matrix)
        - e.g. <[], float32> (scalar float)
        - e.g. [None, L], int32 (minibatch of seqs with unknown batch size)
    - Constatns
        - B := batch size
        - L := length of the longest sequence in batch (tf.shape(batch)[1])
        - E := embedding dimensions i.e. model size.
        - V := vocabulary size
    - Nested Structure
        - Typescript-style notation is used to describe the structure of
        Tensorflow's "nested structure" (nested list, dict (+tuple)).
"""

INF = 1e10


def positional_encoding(length, d_model):
    """Sinusoidal Positional Encoding
    Args:
        length: sentence length (batch.shape[1])
        d_model: embedding size (batch.shape[-1])

    Returns:
        positional_encoding of shape [seq_length, d_model]
    """
    # PE(pos, i) = 
    #   sin(pos/(10000^(i/(d_model/2)))) (0<=i<d_model/2)
    #   cos(pos/(10000^(i/(d_model/2)))) (d_model/2<=i<d_model)
    pos = tf.range(tf.cast(length, tf.float32))
    half = d_model // 2
    i = tf.range(half, dtype=tf.float32)
    scaled_time = pos[:, None] / tf.pow(10000.0, i / half)[None]
    return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


def seq_to_mask(seq, dtype=tf.int32, pad_value=0):
    """
    Args:
        seq: [Batch, MaxL]. 0 for padding.
        dtype: int | float | bool
    Returns:
        <[Batch, MaxL], dtype>
        0 for paddings, 1 for the rest.
    """
    return tf.cast(seq != pad_value, dtype)


def seq_to_padding_bias(seq, dtype=tf.float32, pad_value=0):
    """
    Returns:
        <[Batch, 1, 1, MaxL], dtype>
        -INF for paddings, 0 for the rest.
    """
    return -INF * (1 - seq_to_mask(seq, tf.float32, pad_value)[:, None, None])


def create_padding_bias(lengths, maxlen):
    """
    Args:
        lengths: [batch], tf.int32.
        maxlen: int32 | None
    returns:
        [batch, 1, 1, maxlen], float32.
        -INF for paddings.
    """
    mask = tf.sequence_mask(lengths, maxlen, tf.float32)
    return ((1 - mask) * (-INF))[:, None, None]


def create_causal_mask(length):
    """Lower triangular matrix filled with 1.
    Args:
        length: size of the mask matrix
    Returns:
        [1, 1, length, length], float32.
        """
    mat = tf.sequence_mask(tf.range(length) + 1, length, tf.float32)
    return mat[None, None]


def create_causal_bias(length):
    """
    Args:
        length: size of the mask matrix
    Returns:
        [1, 1, length, length], float32.
        Upper triangular part excluding the diagonal are filled
        with -INF
    """
    return (1 - create_causal_mask(length)) * (-INF)


def transfer_padding_to_left(seq, pad=0):
    """
    Args:
        seq: <[B, L], int32> batch of sequneces with paddings at the right
    Returns:
        (new_seq: <[B, L], int32>, offsets: <[B], int32>)
    """
    L = tf.shape(seq)[1]
    offsets = tf.reduce_sum(tf.cast(seq == pad, tf.int32), axis=1)
    indices = tf.math.maximum(-1, tf.range(L) - offsets[:, None]) % L
    new_seq = tf.gather(seq, indices, batch_dims=1)
    return new_seq, offsets


def transfer_padding_to_right(seq, pad=0, offsets=None):
    """
    Args:
        seq: <[B, L], int32> batch of sequneces with paddings at the left
    Returns:
        new_seq: <[B, L], int32>
    """
    L = tf.shape(seq)
    if offsets is None:
        offsets = tf.reduce_sum(tf.cast(seq == pad, tf.int32), axis=1)
    indices = tf.math.minimum(L, tf.range(L) + offsets[:, None]) % L
    new_seq = tf.gather(seq, indices, batch_dims=1)
    return new_seq


def label_smoothing(labels, eps=0.1):
    if eps == 0:
        return labels
    else:
        V = tf.cast(tf.shape(labels)[-1], tf.float32)
        return (1 - eps) * labels + eps / V


class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, input_dim, output_dim, scale, **kwargs):
        super().__init__(**kwargs)

        self.emb = self.add_weight(
            name='emb',
            shape=[input_dim, output_dim],
            dtype=tf.float32)
        
        self.scale = scale


    def call(self, inputs):
        outputs = tf.gather(self.emb, inputs)

        if self.scale:
            dims = tf.cast(tf.shape(self.emb)[1], tf.float32)
            outputs = outputs * (dims ** 0.5)

        return outputs


    def emb2logits(self, inputs):
        # These two are equivalent but `tensordot` might be faster
        # return tf.matmul(inputs, self.emb[None], transpose_b=True)
        return tf.tensordot(inputs, self.emb, [[-1],[1]])


class MultiheadAttention(keras.layers.Layer):
    def __init__(self, d_model, n_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.q = keras.layers.Dense(self.d_model, use_bias=False, name='q')
        self.k = keras.layers.Dense(self.d_model, use_bias=False, name='k')
        self.v = keras.layers.Dense(self.d_model, use_bias=False, name='v')
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.out = keras.layers.Dense(self.d_model, use_bias=False, name='out')


    def call(self, query, target, bias, training, cache=None):
        """
        Args:
            query: [B, len_q, E]
            target: [B, len_t, E]
            bias: [(1|B), 1, len_q, len_t], float32
            cache: {
                'k': Tensor<[B, L_cache, E], float32>,
                'v': Tensor<[B, L_cache, E], float32>
            }
        """
        head_size = self.d_model // self.n_heads
        
        q = self.q(query) # [batch, length, d_model]
        k = self.k(target)
        v = self.v(target)

        if cache is not None:
            k = tf.concat([cache['k'], k], axis=1)
            v = tf.concat([cache['v'], v], axis=1)
            cache['k'] = k
            cache['v'] = v

        # [batch, nheads, length_q, head_size]
        q = tf.stack(tf.split(q, self.n_heads, axis=-1), axis=1)
        # [batch, nheads, length_k, head_size]
        k = tf.stack(tf.split(k, self.n_heads, axis=-1), axis=1)
        v = tf.stack(tf.split(v, self.n_heads, axis=-1), axis=1)

        # [batch, nheads, length_q, length_k]
        weight = tf.matmul(q, k, transpose_b=True)
        weight = weight / (head_size ** 0.5)

        # Masking (e.g. padding/causal mask)
        weight = weight + bias

        weight = tf.nn.softmax(weight, name='attention_weight')
        weight = self.dropout(weight, training=training)

        # [batch, nheads, length_q, head_size]
        outputs = tf.matmul(weight, v)

        # [batch, length_q, emb_size]
        outputs = tf.concat(tf.unstack(outputs, axis=1), axis=2)

        outputs = self.out(outputs)

        return outputs


    def create_cache(self, batch_size):
        """
        Returns:
            (cache: dict, shape_invariant: dict)
        """
        k = tf.zeros([batch_size, 0, self.d_model])
        v = tf.zeros([batch_size, 0, self.d_model])
        shape = tf.TensorShape([None, None, self.d_model])
        return {'k': k, 'v': v}, {'k': shape, 'v': shape}
    

    def permute_cache(self, cache, ids):
        for x in ('k', 'v'):
            cache[x] = tf.gather(cache[x], ids)


class SelfAttention(MultiheadAttention):
    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, inputs, *args, **kwargs)


class Feedforward(keras.layers.Layer):
    def __init__(self, n_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_units = n_units
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.relu = keras.layers.Dense(
            n_units, activation='relu', use_bias=True, name='relu')


    def build(self, input_shape):
        self.linear = keras.layers.Dense(
            input_shape[-1], use_bias=True, name='linear')


    def call(self, inputs, training=False):
        # This dropout is not written in the original paper.
        # I don't remember why but it was here also in the first version
        # of this code.
        outputs = self.relu(inputs)
        outputs = self.dropout(outputs, training=training)
        return self.linear(outputs)


class NormResidualWrapper(keras.layers.Layer):
    def __init__(
        self, layer_constructor, dropout_rate, norm_eps, **kwargs):
        """
        Wraps the given layer (usually MH-attention or feed-foward)
        with layer-norm, dropout, and residual connection.
        Args:
            layer_constructor:
                A callable which creates the sublayer instance. Instanciating
                the sublayer in this __init__ is important because it enables
                this class to track the weights of the sublayer.
        """
        super().__init__(**kwargs)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=norm_eps)
        self.layer = layer_constructor()
        self.dropout = keras.layers.Dropout(dropout_rate)


    def call(self, x, *args, training, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, training=training, **kwargs)
        y = self.dropout(y, training=training)
        return y + x


class EncoderBlock(keras.layers.Layer):
    def __init__(
            self, d_model, n_heads, dropout_rate, ff_size, norm_eps, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = NormResidualWrapper(
            lambda: SelfAttention(d_model, n_heads, dropout_rate),
            dropout_rate,
            norm_eps)
        self.ff = NormResidualWrapper(
            lambda: Feedforward(ff_size, dropout_rate),
            dropout_rate,
            norm_eps)
    

    def call(self, x, bias, training):
        y = self.self_attn(x, bias, training=training)
        return self.ff(y, training=training)
        

class EncoderRelPosBlock(EncoderBlock):
    def __init__(
            self, d_model, n_heads, dropout_rate, ff_size, norm_eps,
            rel_pos_max_dist, rel_pos_unique_per_head,
            context=True, **kwargs):
        super().__init__(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate,
            ff_size=ff_size, norm_eps=norm_eps, **kwargs)
        self.self_attn = NormResidualWrapper(
            lambda: RelativePositionMultiheadSelfAttention(
                d_model, n_heads, dropout_rate,
                rel_pos_max_dist, rel_pos_unique_per_head),
            dropout_rate,
            norm_eps)


class Encoder(keras.layers.Layer):
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
        use_rel_pos,
        rel_pos_max_dist=None,
        rel_pos_unique_per_head=None,
        in_emb=None,
        **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        if in_emb is not None:
            self.in_emb = in_emb
        else:
            self.in_emb = EmbeddingLayer(vocab_size, d_model, scale=True)
        if use_pos_enc: self.pos_enc = positional_encoding(maxlen, d_model)
        if use_pos_emb: self.pos_emb = self.add_weight(
            name='pos_emb', shape=[maxlen, d_model], dtype=tf.float32)
        self.post_emb_dropout = keras.layers.Dropout(dropout_rate)
        if not use_rel_pos:
            self.blocks = [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout_rate=dropout_rate,
                    ff_size=ff_size,
                    norm_eps=norm_eps, 
                    name=f'layer_{i}')
                for i in range(n_blocks)
            ]
        else:
            assert rel_pos_max_dist is not None\
                and rel_pos_unique_per_head is not None
            self.blocks = [
                EncoderRelPosBlock(
                    d_model, n_heads, dropout_rate, ff_size, norm_eps,
                    rel_pos_max_dist, rel_pos_unique_per_head,
                    name=f'layer_{i}')
                for i in range(n_blocks)
            ]
        self.output_norm = keras.layers.LayerNormalization(epsilon=norm_eps)


    def call(self, x, self_attn_bias, training):
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
            y = block(y, self_attn_bias, training=training)

        return self.output_norm(y)


class DecoderBlock(keras.layers.Layer):
    def __init__(
            self, d_model, n_heads, dropout_rate, ff_size, norm_eps,
            context=True, **kwargs):
        super().__init__(**kwargs)
        # Self-attention layer
        self.self_attn = NormResidualWrapper(
            lambda: SelfAttention(d_model, n_heads, dropout_rate),
            dropout_rate,
            norm_eps)

        # Dec-Enc attention layer
        if context:
            self.ctx_attn = NormResidualWrapper(
                lambda: MultiheadAttention(d_model, n_heads, dropout_rate),
                dropout_rate,
                norm_eps)
        else:
            self.ctx_attn = None

        # Feedforward layer
        self.ff = NormResidualWrapper(
            lambda: Feedforward(ff_size, dropout_rate),
            dropout_rate,
            norm_eps)
    

    def call(
            self, x, self_attn_bias, training,
            ctx=None, ctx_attn_bias=None, cache=None):
        y = self.self_attn(x, self_attn_bias, training=training, cache=cache)
        if self.ctx_attn is not None:
            assert ctx is not None and ctx_attn_bias is not None
            y = self.ctx_attn(y, ctx, ctx_attn_bias, training=training)
        y = self.ff(y, training=training)
        return y
    

    def create_cache(self, batch_size):
        return self.self_attn.layer.create_cache(batch_size)
    

    def permute_cache(self, cache, ids):
        self.self_attn.layer.permute_cache(cache, ids)


class DecoderRelPosBlock(DecoderBlock):
    def __init__(
            self, d_model, n_heads, dropout_rate, ff_size, norm_eps,
            rel_pos_max_dist, rel_pos_unique_per_head,
            context=True, **kwargs):
        super().__init__(
            d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate,
            ff_size=ff_size, norm_eps=norm_eps, context=context, **kwargs)
        # Override Self-attention layer
        self.self_attn = NormResidualWrapper(
            lambda: RelativePositionMultiheadSelfAttention(
                d_model, n_heads, dropout_rate,
                rel_pos_max_dist, rel_pos_unique_per_head),
            dropout_rate,
            norm_eps)
    

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
            use_rel_pos,
            n_blocks,
            norm_eps,
            rel_pos_max_dist,
            rel_pos_unique_per_head,
            in_emb=None,
            context=True,
            **kwargs):
        super().__init__(**kwargs)
        if in_emb is not None:
            self.in_emb = in_emb
        else:
            self.in_emb = EmbeddingLayer(vocab_size, d_model, scale=True)
        
        if use_pos_enc: self.pos_enc = positional_encoding(maxlen, d_model)
        if use_pos_emb: self.pos_emb = self.add_weight(
            name='pos_emb', shape=[maxlen, d_model], dtype=tf.float32)

        self.post_emb_dropout = keras.layers.Dropout(dropout_rate)

        if not use_rel_pos:
            self.blocks = [
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout_rate=dropout_rate,
                    ff_size=ff_size,
                    norm_eps=norm_eps, 
                    context=context,
                    name=f'layer_{i}')
                for i in range(n_blocks)
            ]
        else:
            assert rel_pos_max_dist is not None\
                and rel_pos_unique_per_head is not None
            self.blocks = [
                DecoderRelPosBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout_rate=dropout_rate,
                    ff_size=ff_size,
                    norm_eps=norm_eps,
                    rel_pos_max_dist=rel_pos_max_dist,
                    rel_pos_unique_per_head=rel_pos_unique_per_head,
                    context=context,
                    name=f'layer_{i}')
                for i in range(n_blocks)
            ]
        
        self.output_norm = keras.layers.LayerNormalization(epsilon=norm_eps)


    def call(
            self, x,
            self_attn_bias,
            training,
            context=None, ctx_attn_bias=None,
            cache=None,
            offsets=None,
            ret_embedding=False):
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
            self_attn_bias = create_causal_bias(r)[:, :, l:]

        # Adjust the position-related tensors
        if offsets is None:
            pos_enc = self.pos_enc[l: r] if hasattr(self, 'pos_enc') else None
            pos_emb = self.pos_emb[l: r] if hasattr(self, 'pos_emb') else None
        else:
            # [B, 1, 1, r] offset bias
            offset_bias = -INF * tf.sequence_mask(
                offsets, r, dtype=tf.float32)[:, None, None]
            # [B, 1, r - l, r]
            self_attn_bias += offset_bias

            # [ML, E] -> [B, r-l, E]
            indices = tf.range(l, r)[None] - offsets[:, None]
            pos_enc = tf.gather(self.pos_enc, indices) \
                if hasattr(self, 'pos_enc') else None
            pos_emb = tf.gather(self.pos_emb, indices) \
                if hasattr(self, 'pos_emb') else None


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
                training=training,
                ctx=context,
                ctx_attn_bias=ctx_attn_bias,
                cache=cache[f'layer_{i}'] if cache is not None else None
            )
        
        y = self.output_norm(y)

        if ret_embedding:
            return y
        else:
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


class Transformer(keras.layers.Layer):
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
            use_pos_enc,
            use_pos_emb,
            use_rel_pos,
            share_enc_dec_embedding,
            rel_pos_max_dist=None,
            rel_pos_unique_per_head=None,
            **kwargs):
        super().__init__(**kwargs)

        NORM_EPS = 1e-6 # epsilon of layer norm

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.maxlen = maxlen
        self.ff_size = ff_size
        self.dropout_rate = dropout_rate
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.use_pos_enc = use_pos_enc
        self.use_pos_emb = use_pos_emb
        self.use_rel_pos = use_rel_pos
        self.share_enc_dec_embedding = share_enc_dec_embedding
        self.rel_pos_max_dist = rel_pos_max_dist
        self.rel_pos_unique_per_head = rel_pos_unique_per_head

        self.encoder = Encoder(
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
            use_rel_pos=use_rel_pos,
            rel_pos_max_dist=rel_pos_max_dist,
            rel_pos_unique_per_head=rel_pos_unique_per_head,
            name='encoder')

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
            n_blocks=n_dec_blocks,
            norm_eps=NORM_EPS,
            in_emb=self.encoder.in_emb if share_enc_dec_embedding else None,
            rel_pos_max_dist=rel_pos_max_dist,
            rel_pos_unique_per_head=rel_pos_unique_per_head,
            context=True,
            name='decoder')
    

    def call(
            self, enc_input, dec_input, training,
            ret_embedding=False, offsets=None):
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
        enc_pad_bias = seq_to_padding_bias(enc_input)
        enc_out = self.encoder(
            enc_input, self_attn_bias=enc_pad_bias, training=training)

        
        dec_out = self.decoder(
            dec_input,
            self_attn_bias='causal_bias',
            training=training,
            context=enc_out,
            ctx_attn_bias=enc_pad_bias,
            cache=None,
            offsets=offsets,
            ret_embedding=ret_embedding)
        
        return dec_out
    

    def encode(self, x, training):
        enc_pad_bias = seq_to_padding_bias(x)
        enc_out = self.encoder(
            x, self_attn_bias=enc_pad_bias, training=training)
        return enc_out, enc_pad_bias
    

    def create_cache(self, batch_size):
        """cache and the corresponding shape invariant (both nested)"""
        return self.decoder.create_cache(batch_size)
    

    def create_decoder(
            self, enc_out, enc_pad_bias, init_cache=None, offsets=None):
        def f(dec_input):
            return self.decoder(
                dec_input,
                self_attn_bias='causal_bias',
                training=False,
                context=enc_out,
                ctx_attn_bias=enc_pad_bias,
                cache=init_cache,
                offsets=offsets,
                ret_embedding=False)
        
        return f


    def permute_cache(self, cache, permutation):
        self.decoder.permute_cache(cache, permutation)


    def beam_search_decode_with_prefix(
            self, x, prefix_or_sos, eos, beam_size,
            maxlen=None,
            length_penalty_fn=None):
        """
        Args:
            x: [B, L_enc]
            prefix_or_sos: <[B, L_prefix]> | <[B]> | <[]>
            maxlen: None | <[B]> | <[]> | int
        """
        enc_pad_bias = seq_to_padding_bias(x)
        # [B, L, E]
        enc_out = self.encoder(x, self_attn_bias=enc_pad_bias, training=False)
        B, L, E = (tf.shape(enc_out)[i] for i in range(3))
        K = beam_size

        prefix_or_sos = tf.convert_to_tensor(prefix_or_sos)
        if len(prefix_or_sos.get_shape().as_list()) == 2:
            # [B, L_prefix], [B]
            prefix, offsets = transfer_padding_to_left(prefix_or_sos)
            sos = prefix[:, -1]
        else:
            prefix, offsets = None, None
            sos = tf.broadcast_to(prefix_or_sos, [B])

        maxlen = self.maxlen if maxlen is None \
            else tf.math.minimum(self.maxlen, maxlen)

        # [B * K, L, E]
        rep_enc_out = tf.repeat(enc_out, K, axis=0)

        # [B * K, ...]
        rep_enc_pad_bias = tf.repeat(enc_pad_bias, K, axis=0)

        # [B * K]
        rep_offsets = None if offsets is None else tf.repeat(offsets, K, axis=0)

        # State variables
        cache, shape_inv = self.decoder.create_cache(B * K)

        def get_logits_fn(x, dummy=False):
            logits = self.decoder(
                x,
                self_attn_bias='causal_bias',
                training=False,
                context=rep_enc_out,
                ctx_attn_bias=rep_enc_pad_bias,
                cache=cache,
                offsets=rep_offsets
            )
            if not dummy:
                return tf.nn.log_softmax(logits)
        
        def perm_batch_fn(batch_permutation):
            """
            x: <[B * K], int32>
            """
            self.decoder.permute_cache(cache, batch_permutation)

        def get_state_fn():
            return cache

        def put_controlled_state_fn(cache_):
            nonlocal cache
            cache = cache_

        # Initial call of decoder with the prefixes
        if prefix is not None:
            get_logits_fn(prefix[:, :-1], dummy=True)
        
        # [B, K, L_out], [B, K]
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
            length_penalty_fn=length_penalty_fn
        )

        # [B, K, L_pfx - 1 + L_out]
        # Be aware the last token of the prefix is the first token of the path
        if prefix is not None:
            paths = tf.concat(
                [tf.tile(prefix[:, None, :-1], [1, K, 1]), paths], axis=2)
        
        return paths, scores

    def sample_one(
            self, x, sos, eos, maxlen=None, T=1.0):
        """
        Args:
            x: [B, L_enc]
            prefix_or_sos: <[B]> | 
            maxlen: None | <[B]> | <[]> | int
        """
        enc_pad_bias = seq_to_padding_bias(x)

        # [B, L, E]
        enc_out = self.encoder(x, self_attn_bias=enc_pad_bias, training=False)
        B, L, E = (tf.shape(enc_out)[i] for i in range(3))

        maxlen = self.maxlen if maxlen is None \
            else tf.math.minimum(self.maxlen, maxlen)

        # State variables
        cache, shape_inv = self.decoder.create_cache(B)

        def get_logits_fn(x):
            logits = self.decoder(
                x,
                self_attn_bias='causal_bias',
                training=False,
                context=enc_out,
                ctx_attn_bias=enc_pad_bias,
                cache=cache
            )

            return logits / T
        
        def get_state_fn():
            return cache

        def put_controlled_state_fn(cache_):
            nonlocal cache
            cache = cache_
        
        # [B, L_out]
        paths = sample_one(
            get_logits_fn=get_logits_fn,
            sos=sos,
            eos=eos,
            maxlen=maxlen,
            pad=0,
            get_state_fn=get_state_fn,
            put_controlled_state_fn=put_controlled_state_fn,
            shape_invariants=shape_inv,
        )

        return paths
    

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create an instance with configuration defined in `transformer_net.ts`
        """
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
            use_pos_enc=c['use_pos_enc'],
            use_pos_emb=c['use_pos_emb'],
            use_rel_pos=c['use_rel_pos'],
            share_enc_dec_embedding=c['share_enc_dec_embedding'],
            rel_pos_max_dist=c['rel_pos_max_dist'],
            rel_pos_unique_per_head=c['rel_pos_unique_per_head'],
            **kwargs)
