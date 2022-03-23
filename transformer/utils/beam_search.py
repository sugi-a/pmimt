import sys
from logging import getLogger; logger = getLogger(__name__)
import tensorflow as tf
from tensorflow import keras
import numpy as np

INF = 1e9
BCT = tf.broadcast_to
TShape = tf.TensorShape

def length_penalty(length, alpha):
    """
    Args:
        length: Tensor<any, int32>
        alpha: float
    Returns:
        [shape(length), float32]
    """
    a = tf.cast((5 + length)/(1 + 5), tf.float32)
    b = tf.cast(alpha, tf.float32)
    return tf.cast(tf.pow(a, b), tf.float32)


def create_length_penalty_fn(alpha):
    return lambda length: length_penalty(length, alpha)


def beam_search(
        get_logits_fn,
        perm_batch_fn,
        sos,
        eos,
        beam_size,
        maxlen,
        pad=0,
        get_state_fn=None,
        put_controlled_state_fn=None,
        shape_invariants=None,
        length_penalty_fn=None
        ):
    """
    Primitive beam search for autoregressive sequence generation models.
    Mini-batch inputs are supported.
    B := batch size
    K := beam size
    V := vocabulary size
    Args:
        get_logits_fn:
            (dec_input: <[B * K, 1], int32>) => <[B * K, l, V], float32>
            In:
                Pseudo mini-batch consisting of B * K token IDs.
                The i-th element (0 <= i < B * K) is the newest token
                of the (i % K)-th path of the (i // K)-th batch.
            Out:
                Output is the token scores over the vocabulary, which must be
                log-scale score (normalized or unnormalized logits).
                Sequence score is computed as a sum of the token scores.
        perm_batch_fn:
            (alive_path_ids: <[B * K], int32> => void)
        sos: Integer[B] | <[B], int32>
        maxlen: Integer | Tensor<([]|[B]), int32>
        get_state_fn: ()=>Nested_state<Tensor>
        put_controlled_state_fn: (controlled_state: Nested_state<Tensor>)=>void
        shape_invariants:
            List<Tuple<<Tensor<any>, TensorShape>>> | None
        length_penalty_fn: callable | None
            callable: (length: <any, int32>) => <shape(length), float32>
    Returns:
        paths: <[B, K], int32>, score: <[B, K], float32>
    """
    B = tf.shape(sos)[0]
    K = beam_size

    if length_penalty_fn is None:
        length_penalty_fn = lambda x: 1

    ln_one_of_K = tf.concat([[0.0], tf.fill([K - 1], -INF)], axis=0)

    # [B, K, 1] <- [B]
    paths = BCT(sos[:, None, None], [B, K, 1])
    # [B, K] Sequence log probability
    slogp = BCT(ln_one_of_K[None], [B, K])
    # [B, K] Sequence score (=slogp if no length penalty is used)
    score = tf.identity(slogp)
    # [B, K]
    closed = tf.fill([B, K], False)
    # [B]
    maxlen = BCT(maxlen, [B])
    i = tf.constant(0)

    # External state (if exists)
    assert (get_state_fn is None) == (put_controlled_state_fn is None)
    ex_state = None if get_state_fn is None else get_state_fn()

    shape_inv = [(ex_state, shape_invariants), (paths, TShape([None, None, None]))]

    while ~tf.math.reduce_all(closed):
        tf.autograph.experimental.set_loop_options(shape_invariants=shape_inv)
        if put_controlled_state_fn is not None:
            put_controlled_state_fn(ex_state)

        # [B * K, V]
        t_logp = get_logits_fn(tf.reshape(paths, [B * K, -1])[:, -1:])[:, 0]
        V = tf.shape(t_logp)[1]
        # [B, K, V]
        t_logp = tf.reshape(t_logp, [B, K, -1])

        # Force EOS for sequences longer than or equal to their maxlen
        non_eos_bias = tf.concat([
            tf.ones_like(t_logp[:, :, :eos], tf.float32) * (-INF),
            t_logp[:, :, eos: eos + 1],
            tf.ones_like(t_logp[:, :, eos + 1:], tf.float32) * (-INF)
        ], axis=-1)
        t_logp = tf.where(i + 1 >= maxlen[:, None, None], non_eos_bias, t_logp)

        # Set logp=0 for already closed paths
        ln_one_of_V = tf.concat([[0.0], tf.fill([V - 1], -INF)], axis=0)
        t_logp = tf.where(closed[:, :, None], ln_one_of_V[None, None], t_logp)

        # new sequence logp and score
        t_slogp = slogp[:, :, None] + t_logp

        # Apply length penalty
        t_score = tf.where(
            closed[:, :, None],
            score[:, :, None],
            t_slogp / length_penalty_fn(i + 1)
        )

        # [B, K, V] -> [B, K * V] -> [B, K] Top K.
        top_score, top_indices = tf.math.top_k(
            tf.reshape(t_score, [B, -1]), k=K, sorted=False)

        # [B, K]
        # 0 <= x < K
        alive_path_ids = top_indices // tf.shape(t_score)[-1]
        # 0 <= x < V
        new_token_ids = top_indices % tf.shape(t_score)[-1]


        # Update loop states
        old_close = tf.gather(closed, alive_path_ids, batch_dims=1)
        perm_batch_fn(
            tf.reshape(alive_path_ids + tf.range(B)[:, None] * K, [-1]))

        # [B, K, L+1] <- [B, K, L]
        paths = tf.concat([
                tf.gather(paths, alive_path_ids, batch_dims=1),
                tf.where(old_close, pad, new_token_ids)[:, :, None]
            ], axis=2)

        # [B, K] <- [B, K*V] <- [B, K, V]
        slogp = tf.gather(
            tf.reshape(t_slogp, [B, -1]), top_indices, batch_dims=1)
        score = top_score
        closed = old_close | (new_token_ids == eos)
        i += 1

        # Update external state (if exists)
        if ex_state is not None:
            ex_state = get_state_fn()


    # Sort
    score, indices = tf.math.top_k(score, K, sorted=True)
    paths = tf.gather(paths, indices, batch_dims=1)

    return paths, score


def sample_one(
    get_logits_fn,
    sos,
    eos,
    maxlen,
    pad=0,
    get_state_fn=None,
    put_controlled_state_fn=None,
    shape_invariants=None,
    ):
    
    B = tf.shape(sos)[0]

    # [B, 1] <- [B]
    paths = BCT(sos[:, None], [B, 1])

    # [B]
    closed = tf.fill([B], False)

    # [B]
    maxlen = BCT(maxlen, [B])

    i = tf.constant(0)

    # External state (if exists)
    assert (get_state_fn is None) == (put_controlled_state_fn is None)
    ex_state = None if get_state_fn is None else get_state_fn()

    shape_inv = [(ex_state, shape_invariants), (paths, TShape([None, None]))]

    while ~tf.math.reduce_all(closed):
        tf.autograph.experimental.set_loop_options(shape_invariants=shape_inv)
        if put_controlled_state_fn is not None:
            put_controlled_state_fn(ex_state)

        # [B, V]
        t_logp = get_logits_fn(paths[:, -1:])[:, 0]
        V = tf.shape(t_logp)[1]

        # [B, V] -> [B, 1]. Sampling
        samples = tf.random.categorical(t_logp, 1)
        samples = tf.cast(samples, dtype=tf.int32)

        # [B, 1] Decide the new token
        new_token = tf.where(
            closed[:, None], pad,
            tf.where(i + 1 >= maxlen[:, None], eos, samples))

        # [B, L+1] <- [B, L]
        paths = tf.concat([ paths, new_token ], axis=1)

        # [B]
        closed = closed | (new_token[:, 0] == eos) | (new_token[:, 0] == pad)
        
        i += 1

        # Update external state (if exists)
        if ex_state is not None:
            ex_state = get_state_fn()

    return paths
