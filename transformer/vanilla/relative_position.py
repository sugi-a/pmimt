import tensorflow as tf
from tensorflow import keras


class RelativePositionMultiheadSelfAttention(keras.layers.Layer):
    def __init__(
            self, d_model, n_heads, dropout_rate,
            max_relative_dist, unique_per_head, **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_dist = max_relative_dist
        self.unique_per_head = unique_per_head

        self.q = keras.layers.Dense(self.d_model, use_bias=False, name='q')
        self.k = keras.layers.Dense(self.d_model, use_bias=False, name='k')
        self.v = keras.layers.Dense(self.d_model, use_bias=False, name='v')
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.out = tf.layers.Dense(self.d_model, use_bias=False, name='out')

        if unique_per_head:
            # [vocab, nheads, head_size]
            self.pos_emb = self.add_weight(
                shape=[max_relative_dist * 2 + 1, n_heads, d_model // n_heads],
                dtype=tf.float32)
        else:
            # [vocab, head_size]
            self.pos_emb = self.add_weight(
                shape=[max_relative_dist * 2 + 1, d_model // n_heads],
                dtype=tf.float32)
        

    def call(self, query, bias, training, cache=None):
        head_size = self.d_model // self.n_heads

        # [B, L, E]
        q = self.q(query)
        k = self.k(query)
        v = self.v(query)
        
        if cache is not None:
            k = tf.concat([cache['k'], k], axis=1)
            v = tf.concat([cache['v'], v], axis=1)
            cache['k'] = k
            cache['v'] = v

        # [B, nheads, L_q, H]
        q = tf.stack(tf.split(q, self.n_heads, axis=-1), axis=1)
        # [B, nheads, L_k, H]
        k = tf.stack(tf.split(k, self.n_heads, axis=-1), axis=1)
        v = tf.stack(tf.split(v, self.n_heads, axis=-1), axis=1)

        # Normal q-k multiplication term
        # [B, nheads, L_q, L_k]
        weight = tf.matmul(q, k, transpose_b=True)

        # [q_len, k_len] Relative position matrix
        L_q, L_k = tf.shape(q)[2], tf.shape(k)[2]
        k_indices = tf.range(L_k)
        q_indices = k_indices[-L_q:]
        rel_pos = k_indices[None, :] - q_indices[:, None]

        # Clipping
        rel_pos = tf.clip_by_value(rel_pos, -self.max_dist, self.max_dist)

        # Shift to start from 0
        rel_pos += self.max_dist

        # Embedding matrix.
        # [L_q, L_k, nheads, head_size] If unique_per_head
        # [L_q, L_k, head_size] Otherwise
        embeddings = tf.gather(self.pos_emb, rel_pos)

        # Make bias. [batch, nheads, L_q, L_k]
        if self.unique_per_head:
            # q: [batch, nheads, L_q, head], emb: [L_q, L_k, nheads, head]
            rel_pos_bias = tf.einsum('bnqh,qknh->bnqk', q, embeddings)
        else:
            # q: [batch, nheads, L_q, head], emb: [L_q, L_k, head]
            rel_pos_bias = tf.einsum('bnqh,qkh->bnqk', q, embeddings)

        # Apply relative postion bias
        weight += rel_pos_bias

        weight = weight / (head_size ** 0.5)
        weight = weight + bias

        weight = tf.nn.softmax(weight, name='attention_weight')
        weight = self.dropout(weight, training=training)

        # [batch, nheads, length_q, head_size]
        outputs = tf.matmul(weight, v)

        # [batch, length_q, emb_size]
        outputs = tf.concat(tf.unstack(outputs, axis=1), axis=2)

        outputs = self.out(outputs)

        return outputs

