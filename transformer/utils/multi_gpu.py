from logging import getLogger; logger = getLogger(__name__)
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
import numpy as np

"""
Methods related to multi-gpu computation
"""

class PseudoList:
    def __init__(self, lst):
        self.lst = lst
    

    def at(self, i):
        return self.lst[i]


def non_even_split(nested, n):
    """Split each tensor along the 0-th axis
    Args:
        inputs: Nested<Tensor<[B, ...]>>
        n: Integer
        ensure_nonzero_B_by_0_pad:
    Returns:
        list<Nested<Tensor>, n>
    """
    flat = nest.flatten(nested)
    B = tf.shape(flat[0])[0]
    q = B // n
    r = B % n
    partition = tf.concat([tf.fill([r], q + 1), tf.fill([n - r], q)], axis=0)
    split_flat = [tf.split(x, partition, axis=0, num=n) for x in flat]
    return [nest.pack_sequence_as(nested, x) for x in zip(*split_flat)]


def distr_map(fn, inputs_list, devices=None):
    """
    Args:
        inputs_list: list<Nested_IN, N_parallel>
        devices: list<str, N_parallel> | None.
            if None, GPU[0:N_parallel] are used.
    Returns:
        list<Nested_OUT, N_parallel>
        """
    outputs_list = []

    if devices is None:
        devices = [f'/gpu:{i}' for i in range(len(inputs_list))]

    for inputs, device in zip(inputs_list, devices):
        with tf.device(device):
            outputs_list.append(fn(inputs))
    
    return outputs_list


def parallel_map_and_average(fn, inputs_list):
    """
    Args:
        fn: (inputs: Nested_IN<Tensor>)=>(
            outputs: Nested_OUT<Tensor>,
            weight: float)
        inputs_list:
            list<Nested_IN<Tensor>>
    Returns:
        averaged_outputs: Nested_OUT<Tensor>>
        total_weight: float
    """
    out_list, w_list = zip(*distr_map(fn, inputs_list))
    
    with tf.device(None):
        sum_w = tf.add_n(w_list)
        avg_fn = lambda *x: tf.add_n([v * w / sum_w for v,w in zip(x, w_list)])
        out_avg = nest.map_structure(avg_fn, *out_list)

    return out_avg, sum_w


def pad_axis_0(x, n):
    pads = tf.math.maximum(0, n - tf.shape(x)[0])
    pad_shape = tf.concat([pads[None], tf.shape(x)[1:]], axis=0)
    pad = tf.zeros(pad_shape, dtype=x.dtype)
    return tf.concat([x, pad], axis=0)


def split_distr_map_concat(
        fn, inputs, n_parallel,
        ensure_nonzero_B_by_0_pad=False,
        split_device=None, concat_device=None):
    """
    Args:
        fn: (inputs: NestedStrcture_IN<Tensor<[k, any]>)=>
            Nested_OUT<Tensor<[k, any]>
        inputs: Nested_IN<Tensor<[B, any]>
        n_parallel: Integer
    Returns:
        Nested_OUT<Tensor<[B, any]>>
    """
    with tf.device(split_device):
        if ensure_nonzero_B_by_0_pad:
            B_orig = tf.shape(nest.flatten(inputs)[0])[0]
            inputs = nest.map_structure(
                lambda x: pad_axis_0(x, n_parallel), inputs)

        in_list = non_even_split(inputs, n_parallel)    

    out_list = distr_map(fn, in_list)
    
    with tf.device(concat_device):
        conc_fn = lambda *x: tf.concat(x, axis=0)
        outs = nest.map_structure(conf_fn, *out_list)

        if ensure_nonzero_B_by_0_pad:
            outs = nest.map_structure(lambda x: x[:B_orig], outs)
    
    return outs


def get_shape_inv(t):
    rank = len(t.shape.as_list())
    return tf.TensorShape([None] * rank)


def get_spec_shape_inv(t):
    return tf.TensorSpec(get_shape_inv(t), t.dtype)


def list2tensor_array(lst):
    """
    Args:
        lst: list<Nested<Tensor>>
    Returns:
        Nested<TensorArray>
    """
    N = len(lst)
    assert N > 0

    arrays = nest.map_structure(
        lambda x: tf.TensorArray(
            x.dtype,
            size=N,
            infer_shape=False,
            element_shape=get_shape_inv(x)),
        lst[0])
    
    for i, x in enumerate(lst):
        arrays = nest.map_structure(
            lambda a, v: a.write(i, v), arrays, x)
    
    return arrays


def sequential_map_reduce(map_fn, reduce_fn, list_x):
    N = len(list_x)
    assert N > 0

    if N == 1:
        return map_fn(list_x[0])

    map_fn_sig = nest.map_structure(get_spec_shape_inv, list_x[0])
    map_fn = tf.function(map_fn, [map_fn_sig])

    i_arrays = list2tensor_array(list_x[1:])

    outs = map_fn(list_x[0])

    for i in tf.range(N - 1):
        tf.autograph.experimental.set_loop_options(parallel_iterations=1)
        outs = reduce_fn(outs, map_fn(i_arrays.read(i)))

    return outs


def sequential_map(fn, list_x, out_spec):
    """
    Maps each element `x` in `list_x` by `fn(x)` sequentially,
    in which the peak memory consumption is expected to be equal
    to that of a single call of `fn`.

    `tf.TensorArray` and (implicit) `tf.while_loop` is used in this
    implementation. In graph mode, it shows better empirical memory
    efficiency than just calling `fn` for each element of `list_x`
    in the order controlled by `tf.control_dependencies`.
    
    Args:
        fn: (x: Nested_IN<Tensor>)=>Nested_OUT<Tensor>
        x_list: list<Nested_IN<Tensor>, N>
        out_type: Nested_OUT<dtype>
    Returns:
        list<Nested_OUT<Tensor>, N>
    """
    N = len(list_x)
    if N == 0:
        return []
    
    i_arrays = list2tensor_array(list_x)
    o_arrays = nest.map_structure(
        lambda s: tf.TensorArray(
            s.dtype,
            size=N,
            infer_shape=False,
            element_shape=s.shape),
        out_spec)

    for i in tf.range(N):
        tf.autograph.experimental.set_loop_options(parallel_iterations=1)
        x = nest.map_structure(lambda a: a.read(i), i_arrays)
        y = fn(x)
        o_arrays = nest.map_structure(
            lambda a, y_: a.write(i, y_),
            o_arrays, y)
    
    return [nest.map_structure(lambda a: a.read(i), o_arrays) for i in range(N)]


def split_sequential_map_concat(
        fn, inputs, n_split, ensure_nonzero_B_by_0_pad=False):
    """
    Args:
        fn: (inputs: NestedStrcture_IN<Tensor<[k, any]>)=>
            Nested_OUT<Tensor<[k, any]>
        inputs: Nested_IN<Tensor<[B, any]>
        n_split: Integer
    Returns:
        Nested_OUT<Tensor<[B, any]>>
    """
    if ensure_nonzero_B_by_0_pad:
        B_orig = tf.shape(nest.flatten(inputs)[0])[0]
        inputs = nest.map_structure(
            lambda x: pad_axis_0(x, n_parallel), inputs)

    in_list = non_even_split(inputs, n_split)    

    out_list = sequential_map(fn, in_list)
    
    conc_fn = lambda *x: tf.concat(x, axis=0)
    outs = nest.map_structure(conf_fn, *out_list)

    if ensure_nonzero_B_by_0_pad:
        outs = nest.map_structure(lambda x: x[:B_orig], outs)
    
    return outs
