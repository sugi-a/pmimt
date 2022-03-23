from logging import getLogger; logger = getLogger(__name__)
import itertools
import random

import numpy as np

from .vocabulary import Vocabulary

"""
Frequently used notation
- Types
    - Basically, Typescript-like type notation is often used
    - Seq: list<int> | tuple<int>
    - Tok: str token
- Method names
    - gen_*: generator method
"""


def gen_line2IDs(line_iter, vocab):
    """
    Args:
        line_iter: Iter<list<Tok>>
        vocab: Vocabulary
    Returns:
        Yield list<int>
    """
    for line in line_iter:
        yield vocab.line2IDs(line)


def gen_line2IDs_multi(lines_iter, vocabs):
    """
    Args:
        vocabs: Vocabulary[] | Vocabulary.
    """
    if isinstance(vocabs, Vocabulary):
        for lines in lines_iter:
            yield tuple(map(vocabs.line2IDs, lines))
    else:
        for lines in lines_iter:
            yield tuple(vocab.line2IDs(line)
                for vocab, line in zip(vocabs, lines))


def gen_skip_empty_line_multi_all(lines_iterable):
    filter_fn = lambda lines: any(l.strip() != '' for l in lines)
    return filter(filter_fn, lines_iterable)


def list2numpy_nested(nested):
    """Converts lists in the nested structure into numpy arrays
    Args:
        nested: nested structure of tuple and dict. lists are
            considered to be objects.
    Returns:
        The same nested structure as the input where lists are
        converted to numpy arrays.
    """
    if isinstance(nested, dict):
        new = {k: list2numpy_nested(v) for k, v in nested.items()}
    elif isinstance(nested, tuple):
        new = tuple(list2numpy_nested(v) for v in nested)
    elif isinstance(nested, list):
        new = np.array(nested, dtype=int)
    else:
        new = nested

    return new


def gen_list2numpy_nested(nested_iter):
    for nested in nested_iter:
        yield list2numpy_nested(nested)


def pad_seqs(seqs, maxlen=None, PAD_ID=0):
    maxlen = maxlen or max(len(seq) for seq in seqs)
    return [seq + [PAD_ID] * (maxlen - len(seq)) for seq in seqs]


def gen_pad_batch_multi(multi_batch_iter, PAD_ID=0):
    """
    Args:
        batch_iter: Iter<tuple<Batch>>
            Batch: list<Seq>
    Returns:
        Yield tuple<Batch<Seq>>.
        All the sequences in the same batch have the same length.
    """
    for batches in multi_batch_iter:
        yield tuple(map(pad_seqs, batches))


def gen_batch_multi(seqs_iter, batch_size):
    """
    Args:
        batch_iter: Iter<tuple<Seq>>
    Returns:
        Yield tuple<Batch<seq>>
    """
    buf = []
    for seqs in seqs_iter:
        buf.append(seqs)
        if len(buf) == batch_size:
            yield tuple(map(list, zip(*buf)))
            buf = []

    if len(buf) > 0:
        yield tuple(map(list, zip(*buf)))


def gen_batch_of_capacity_multi(
        seqs_iterable, capacity, width_fn=None, capacity_fn=None):
    """
    Args:
        width_fn: (seqs: tuple<Seq>)=>int
        capacity_fn: (width: int, height: int)=>int
    Returns:
        tuple<list<Seq>>
    """
    if capacity_fn is None:
        capacity_fn = lambda ws, h: sum(ws) * h

    seqs_iter = iter(seqs_iterable)
    first = next(seqs_iter)
    nseqs = len(first)

    batch = []
    maxws = [0] * nseqs

    for seqs in itertools.chain([first], seqs_iter):
        ws = tuple(map(len, seqs))
        assert capacity_fn(ws, 1) <= capacity

        maxws_ = tuple(map(max, zip(maxws, ws)))
        if capacity_fn(maxws_, len(batch)) > capacity:
            yield tuple(map(list, zip(*batch)))
            batch = []
            maxws = [0] * nseqs
        
        maxws = tuple(map(max, zip(maxws, ws)))
        batch.append(seqs)
    
    if len(batch) > 0:
            yield tuple(map(list, zip(*batch)))


def gen_padded_batch_multi(
        seqs_iter, batch_size=None,
        capacity=None, width_fn=None, capacity_fn=None,
        PAD_ID=0):
    assert (batch_size is None) != (capacity is None)

    fn = gen_batch_multi if batch_size is not None else \
        lambda a,b: gen_batch_of_capacity_multi(a, b, width_fn, capacity_fn)

    yield from gen_pad_batch_multi(fn(seqs_iter, batch_size), PAD_ID=PAD_ID)


def gen_random_sample(iterable, bufsize=None):
    if bufsize is None:
        iterable = list(iterable)
        yield from random.sample(iterable, len(iterable))
    else:
        buf = []
        for x in iterable:
            if len(buf) < bufsize:
                if len(buf) % 10000 == 0:
                    logger.debug('Filling sampling buffer ({}/{})'.format(len(buf), bufsize))
                buf.append(x)
            else:
                ind = random.randint(0, bufsize - 1)
                yield buf[ind]
                buf[ind] = x
        
        random.shuffle(buf)
        yield from buf


def gen_segment_sort(iterable, segsize=10000, key=None):
    i_seg = []
    o_seg = None
    f = True
    for x in iterable:
        i_seg.append(x)
        if len(i_seg) >= segsize:
            i_seg.sort(key=key, reverse=f)
            f = not f
            o_seg = iter(i_seg)
            i_seg = []
        if o_seg:
            yield next(o_seg)
    if o_seg:
        yield from o_seg
    i_seg.sort(key=key)
    yield from i_seg


def gen_line_from_file(fname):
    with open(fname) as f:
        yield from f


def gen_line_from_files(fname_iter):
    for fname in fname_iter:
        yield from gen_line_from_file(fname)


def gen_line_from_files_multi(fnames_iter):
    for fnames in fnames_iter:
        yield from zip(*map(gen_line_from_file, fnames))


def gen_fold(iterable, n, padding_for_remainder=None):
    iterator = iter(iterable)
    if padding_for_remainder is None:
        while True:
            try:
                yield tuple(next(iterator) for i in range(n))
            except:
                break
    else:
        buf = [None] * n
        while True:
            for i in range(n):
                try:
                    buf[i] = next(iterator)
                except:
                    if i > 0:
                        yield tuple(buf[:i] + [padding_for_remainder] * (n - i))
                    return
            yield tuple(buf) 


def fork_iterable(iterable, n):
    its = itertools.tee(iterable, n)
    return tuple(map(lambda x:x[i], it) for i, it in enumerate(its))


class ChainableGenerator:
    def __init__(self, gen):
        self.gen = gen
    

    def trans(self, fn, *args, **kwargs):
        return ChainableGenerator(lambda: fn(self.gen(), *args, **kwargs))

    
    def map(self, fn):
        return ChainableGenerator(lambda: map(fn, self.gen()))
    

    def map_flat(self, fn):
        """
        Args:
            fn: (element)=>iterable
        """
        return ChainableGenerator(
            lambda: itertools.chain.from_iterable(map(fn, self.gen())))


    def __call__(self):
        return self.gen()


    @classmethod
    def zip(cls, *chainables):
        return cls(lambda: zip(*(g() for g in chainables)))
