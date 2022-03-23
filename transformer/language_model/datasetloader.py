import sys, json, random
from logging import getLogger; logger = getLogger(__name__)
from collections import deque
import itertools
import random

from ..custom_text_data_pipeline import core as dp


def gen_doc_from_lines(seq_iterable):
    doc = []
    for seq in seq_iterable:
        if len(seq) == 0:
            if len(doc) > 0:
                yield doc
            doc = []
        else:
            doc.append(seq)
    if len(doc) > 0:
        yield doc


def create_simple_batch_generator(
        files,
        vocab,
        stochastic,
        batch_capacity,
        shuf_buf_size=None,
        length_smoothing=None,
        batch_shuf_buf_size=None):
    gen = dp.ChainableGenerator(lambda: files)
    
    if stochastic: gen = gen.trans(dp.gen_random_sample)

    gen = gen.trans(dp.gen_line_from_files)
    gen = gen.trans(dp.gen_line2IDs, vocab)
    
    if stochastic:
        gen = gen.trans(dp.gen_random_sample, shuf_buf_size)

    if length_smoothing is not None:
        gen = gen.trans(
            dp.gen_segment_sort,
            segsize=length_smoothing,
            key=len)
    
    gen = gen.map(lambda seq: (seq,))
    gen = gen.trans(dp.gen_batch_of_capacity_multi, batch_capacity)
    gen = gen.trans(dp.gen_pad_batch_multi)
    gen = gen.map(lambda seqs: seqs[0])

    if batch_shuf_buf_size is not None:
        gen = gen.trans(dp.gen_random_sample, batch_shuf_buf_size)
    
    return gen


def gen_front_aligned_segment(
        seq_iterable, max_win, min_win, min_stride, rand_extra_stride=0):
    assert min_stride > 0
    q_tok = deque()
    q_len = deque()
    for seq in seq_iterable:
        if len(seq) == 0:
            while len(q_tok) >= min_win:
                # Yield all
                yield list(q_tok)

                # Throw the front seqs
                thrown = 0
                m_stride_ = min_stride + random.randrange(rand_extra_stride + 1)
                while q_len and thrown < m_stride_:
                    l = q_len.popleft()
                    thrown += l
                    for _ in range(l):
                        q_tok.popleft()
                
            q_tok.clear()
            q_len.clear()
        else:
            q_tok.extend(seq)
            q_len.append(len(seq))
            if len(q_tok) < max_win:
                continue
            while len(q_tok) >= max_win:
                # Yield the front win_size toks
                yield list(itertools.islice(q_tok, max_win))

                # Throw the front seqs
                thrown = 0
                m_stride_ = min_stride + random.randrange(rand_extra_stride + 1)
                while q_len and thrown < m_stride_:
                    l = q_len.popleft()
                    thrown += l
                    for _ in range(l):
                        q_tok.popleft()

    if q_tok:
        yield list(q_tok)


def create_front_aligned_doc_segment_generator(
        files,
        vocab,
        stochastic,
        max_window_size,
        min_window_size,
        min_stride,
        rand_extra_stride,
        capacity,
        shuf_buf_size=None
        ):
    gen = dp.ChainableGenerator(lambda: files)

    if stochastic: gen = gen.trans(dp.gen_random_sample)

    gen = gen.trans(dp.gen_line_from_files)
    gen = gen.trans(dp.gen_line2IDs, vocab)

    gen = gen.trans(
        gen_front_aligned_segment,
        max_win=max_window_size,
        min_win=min_window_size,
        min_stride=min_stride,
        rand_extra_stride=rand_extra_stride)
    
    if stochastic:
        gen = gen.trans(dp.gen_random_sample, shuf_buf_size)

    gen = gen.map(lambda seq: (seq,))
    gen = gen.trans(dp.gen_batch_of_capacity_multi, capacity)
    gen = gen.trans(dp.gen_pad_batch_multi)
    gen = gen.map(lambda seqs: seqs[0])
    return gen


class MultiSentenceSlidingWindowLoader:
    def __init__(self, files, vocab, window_size, keep_remainder_larger_equal=None,
        random=True, state_log_file=None,
        header = None):

        self.files = files
        self.vocab = vocab
        self.window_size = window_size
        self.keep_rem_le = keep_remainder_larger_equal
        self.random = random
        self.state_log_file = state_log_file
        if header is None:
            self.header = None
        else:
            if type(header) == int:
                self.header = header
            elif type(header) == str:
                self.header = self.vocab.tok2ID[header]
            else:
                raise ValueError

    def __call__(self):
        return self.gen()


    def gen(self):
        files = random.sample(self.files, len(self.files))
        if self.state_log_file:
            files = gen_json_resumable(files, self.state_log_file)
        for fn in files:
            logger.debug('Opening file {}'.format(fn))
            with open(fn) as f:
                q = deque()
                win_size = random.randint(1, self.window_size) if self.random else self.window_size
                for line in f:
                    # Check the document boundaries
                    if len(line) == 1:
                        if self.header is not None:
                            q.appendleft(self.header)
                        popped = list(q)
                        q.clear()
                        if len(popped) >= self.keep_rem_le:
                            yield popped
                    else:
                        q.extend(self.vocab.line2IDs(line))

                        while len(q) >= self.window_size:
                            if self.header is not None:
                                q.appendleft(self.header)
                            popped = [q.popleft() for i in range(win_size)]
                            if len(popped) >= self.keep_rem_le:
                                yield popped
                            win_size = self.window_size

                # Yield the remainder
                if self.header is not None:
                    q.appendleft(self.header)
                popped = list(q)
                if len(popped) >= self.keep_rem_le:
                    yield popped

