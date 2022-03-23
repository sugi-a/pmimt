from logging import getLogger; logger = getLogger(__name__)
from logging import DEBUG, INFO, basicConfig
import argparse
import sys
import json
import time

import tensorflow as tf
from tensorflow import nest, keras
from tensorflow.nest import map_structure
import numpy as np

from .layers import Transformer, transfer_padding_to_left
from ..utils.beam_search import length_penalty
from ..custom_text_data_pipeline import core as dp
from ..custom_text_data_pipeline.vocabulary import Vocabulary

TSpec = tf.TensorSpec
TI32 = tf.int32

def get_len_penalty_fn(alpha):
    return lambda l: length_penalty(l, alpha)


def create_mask(y, dtype=tf.float32):
    return tf.cast(y != 0, dtype)


def count_toks(y, dtype=tf.float32):
    return tf.reduce_sum(create_mask, dtype)


def deco_cache_provider(f):
    cache = {}
    def f_(*args):
        return f(*args, cache)
    
    return f_


def is_tensor_spec(nested):
    return all(isinstance(a, TSpec) for a in nest.flatten(nested))


def get_signed_func_wrapper(f, signature):
    tspec_idx = [i for i, s in enumerate(signature) if is_tensor_spec(s)]
    tensor_sig = [signature[idx] for idx in tspec_idx]

    @tf.function(input_signature=tensor_sig)
    def f_(*tensors):
        args = signature[:]
        for tensor, idx in zip(tensors, tspec_idx):
            args[idx] = tensor
        return f(*args)
    
    def wrapper(*args):
        assert all(a is s for i,(a,s) in
            enumerate(zip(args, signature)) if i not in tspec_idx)
        return f_(*[args[idx] for idx in tspec_idx])
    
    return wrapper


class InferenceBase:
    def __init__(
            self,
            transformer_model,
            vocab_source,
            vocab_target,
            batch_capacity):
        
        self.model = transformer_model

        self.vocab_src = vocab_source
        self.vocab_trg = vocab_target
        
        self.batch_capacity = batch_capacity
        
    
    def create_data_gen_multi(self, x_multi, vocabs):
        return (
            dp.ChainableGenerator(lambda: iter(x_multi))
            .trans(dp.gen_line2IDs_multi, vocabs)
            .trans(dp.gen_batch_of_capacity_multi, self.batch_capacity)
            .trans(dp.gen_pad_batch_multi)
        )


    def dataset_from_gen(self, gen, structure):
        dtype = map_structure(lambda x: tf.int32, structure)
        shape = map_structure(lambda x: tf.TensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def create_dataset_multi(self, xs, vocabs):
        gen = self.create_data_gen_multi(zip(*xs), vocabs)
        return self.dataset_from_gen(gen, (None,) * len(vocabs))


    @deco_cache_provider
    def comp_translate(self, x, pfx, beam_size, maxlen_ratio, length_penalty_fn, _cache=None):
        def f(x, pfx, beam_size, maxlen_ratio):
            prefix_or_sos = self.vocab_trg.SOS_ID if pfx is None else pfx
            with tf.device('/gpu:0'):
                if tf.size(x) > 0:
                    src_lens = tf.reduce_sum(create_mask(x), axis=1)
                    maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
                    maxlen = tf.where(src_lens == 0.0, 0, maxlen)
                    # [B, K, L], [B, K]

                    paths, scores = self.model.beam_search_decode_with_prefix(
                        x,
                        prefix_or_sos=prefix_or_sos,
                        eos=self.vocab_trg.EOS_ID,
                        beam_size=beam_size,
                        maxlen=maxlen,
                        length_penalty_fn=length_penalty_fn)
                else:
                    B = tf.shape(x)[0]
                    paths = tf.zeros([B, beam_size, 0], tf.int32)
                    scores = tf.zeros([B, beam_size], tf.float32)

                return paths, scores

        if len(_cache) == 0:
            MF0 = TSpec([], tf.float32)
            M0 = TSpec([], tf.int32)
            M2 = TSpec([None, None], tf.int32)
            _cache[True] = get_signed_func_wrapper(f, [M2, None, M0, MF0])
            _cache[False] = get_signed_func_wrapper(f, [M2, M2, M0, MF0])

        return _cache[pfx is None](x, pfx, beam_size, maxlen_ratio)
    

    @tf.function(input_signature=[
        TSpec([None, None], tf.int32),
        TSpec([], tf.float32),
        TSpec([], tf.float32)])
    def comp_sample_one(self, x, maxlen_ratio, T):
        src_lens = tf.reduce_sum(create_mask(x), axis=1)
        maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
        maxlen = tf.where(src_lens == 0.0, 0, maxlen)

        B = tf.shape(x)[0]

        path = self.model.sample_one(
            x,
            tf.fill([B], self.vocab_trg.SOS_ID),
            self.vocab_trg.EOS_ID,
            maxlen=maxlen,
            T=T)
        return path


    def comp_token_logp(self, x, y_in, y_out):
        with tf.device('/gpu:0'):
            # [B, L-1, V]
            logits = self.model(x, y_in, training=False)
            logp_dist = tf.nn.log_softmax(logits)

            # [B, L-1] <- [B, L-1, V]
            logp = tf.gather(logp_dist, y_out, batch_dims=2)

            return logp * create_mask(y_out)


    @tf.function(input_signature=[TSpec([None, None], TI32)]*2)
    def comp_seq_logp(self, x, y):
        with tf.device('/gpu:0'):
            tok_logp = self.comp_token_logp(x, y[:, :-1], y[:, 1:])
            return tf.reduce_sum(tok_logp, axis=1)
    

    @tf.function(input_signature=[TSpec([None, None], TI32)]*3)
    def comp_seq_logp_conditional(self, x, y, prefix):
        with tf.device('/gpu:0'):
            pfx, offsets = transfer_padding_to_left(prefix)
            L_pfx = tf.shape(pfx)[1]

            # [B, L_pfx + L_y]
            conc = tf.concat([pfx, y], axis=1)
            tok_logp = self.comp_token_logp(x, conc[:, :-1], conc[:, 1:], offsets)
            return tf.reduce_sum(tok_logp[:, L_pfx - 1:], axis=1)
    

    def gen_sents2hypotheses(
            self, x, beam_size, length_penalty=None, prefix=None, maxlen_ratio=2.0):
        """
        Returns:
            yield hypos_and_scores: (str[], float[])
        """
        src_v, trg_v = self.vocab_src, self.vocab_trg

        if prefix is None:
            trans_fn = lambda *b: self.comp_translate(b[0], None, beam_size, maxlen_ratio, length_penalty)
            dataset = self.create_dataset_multi((x,), (src_v,))
            dataset = dataset.map(trans_fn)
        else:
            trans_fn = lambda *b: self.comp_translate(b[0], b[1], beam_size, maxlen_ratio, length_penalty)
            dataset = self.create_dataset_multi((x, prefix), (src_v, trg_v))
            dataset = dataset.map(trans_fn)
        
        dataset = dataset.prefetch(1).unbatch().prefetch(1)

        for hypos, scores in dataset:
            yield trg_v.IDs2text(hypos.numpy()), scores.numpy()


    def gen_sample_one(self, x, maxlen_ratio, T):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        dataset = self.create_dataset_multi((x,), (src_v,))
        dataset = dataset.prefetch(1)
        for batch in dataset:
            trans = self.comp_sample_one(batch[0], maxlen_ratio, T)
            yield from trg_v.IDs2text(trans.numpy())


    def gen_sents2sents(self, x, beam_size=1, length_penalty=None, prefix=None):
        """
        Returns:
            yield str
        """
        for hypos, scores in self.gen_sents2hypotheses(
                x, beam_size, length_penalty, prefix):
            yield hypos[0]
    

    def gen_sents2logps(self, x, y, normalize=False):
        dataset = (
            self.create_dataset_multi((x, y), (self.vocab_src, self.vocab_trg))
            .prefetch(1)
        )
        for x, y in dataset:
            seq_logp = self.comp_seq_logp(x, y)
            if normalize:
                seq_logp /= tf.reduce_sum(create_mask(y[:, 1:]), axis=1)
            for s in seq_logp.numpy():
                yield s


    def sents2ppl(self, x, y):
        def fn_(x, y):
            logp = tf.reduce_sum(self.comp_seq_logp(x, y))
            toks = count_toks(y[:, 1:]) # num of toks to be predicted
            return logp, toks

        dataset = (
            self.create_dataset_multi((x, y), (self.vocab_src, self.vocab_trg))
            .map(fn_).prefetch(1)
        )
        reduce_fn = lambda a,b: (a[0] + b[0], a[1] + b[1])
        logp, toks = dataset.reduce((0.0, 0.0), reduce_fn)
        return np.exp(- logp.numpy() / toks.numpy())


    def gen_sents2conditional_logps(self, x, prefix, y):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        dataset = (
            self.create_dataset_multi((x, y, prefix), (src_v, trg_v, trg_v))
            .prefetch(1)
            .map(lambda *b: self.comp_seq_logp_conditional(b[0], b[1], b[2]))
            .unbatch().prefetch()
        )
        yield from dataset


    def unit_test(self):
        t = time.time()
        with open('./inf_test/dev.ru') as f:
            for i, line in enumerate(self.gen_sents2sents( f, beam_size=1)):
                a = line
                if i % 100 == 0 and i != 0:
                    print(i, (time.time() - t)/i)


def main(argv, in_fp):
    p = argparse.ArgumentParser()
    p.add_argument('--dir', '-d', type=str, default='.')
    p.add_argument('--checkpoint', '--ckpt', type=str)
    p.add_argument('--capacity', type=int, default=16384)
    p.add_argument('--mode', choices=['translate', 'sample', 'ppl', 'logp', 'test'],
        default='translate')
    p.add_argument('-T', type=float, default=1.0)
    p.add_argument('--prefix', action='store_true')
    p.add_argument('--beam_size', type=int, default=1)
    p.add_argument('--length_penalty', type=float)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--debug_eager_function', action='store_true')
    p.add_argument('--progress_report_frequency', '--progress', type=int, default=10**10)
    p.add_argument('--normalize', action='store_true')
    args = p.parse_args(argv)

    if args.debug:
        basicConfig(level=DEBUG)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    # Config
    with open(f'{args.dir}/model_config.json') as f:
        model_config = json.load(f)
    
    with open(f'{args.dir}/vocab_config.json') as f:
        vc = json.load(f)
    
    # Transformer Model
    model = Transformer.from_config(model_config)
    ckpt = tf.train.Checkpoint(model=model)
    if args.checkpoint is None:
        ckpt_path = tf.train.latest_checkpoint(f'{args.dir}/checkpoint_best')
    else:
        ckpt_path = args.checkpoint
    assert ckpt_path is not None
    ckpt.restore(ckpt_path)
    logger.info(f'Checkpoint: {ckpt_path}')

    # Vocabulary
    vocab_src = Vocabulary(
        args.dir + '/' + vc['source_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    vocab_trg = Vocabulary(
        args.dir + '/' + vc['target_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])


    # Inference Class
    inference = InferenceBase(model, vocab_src, vocab_trg, args.capacity)

    if args.mode == 'translate':
        if args.length_penalty is None:
            lp_fn = None
        else:
            lp_fn = get_len_penalty_fn(args.length_penalty)

        if args.prefix:
            x, prefix = fork_iterable((l.split('\t') for l in in_fp), 2)
        else:
            x, prefix = in_fp, None

        t = None
        for i,line in enumerate(inference.gen_sents2sents(
                x,
                beam_size=args.beam_size,
                length_penalty=lp_fn,
                prefix=prefix)):
            print(line)
            if t is None: t = time.time()
            if i > 0 and i % args.progress_report_frequency == 0:
                logger.debug(f'{i}, {(time.time() - t)/i}')
    elif args.mode == 'sample':
        t = None
        for i, line in enumerate(inference.gen_sample_one(in_fp, 2.0, args.T)):
            print(line)
            if t is None: t = time.time()
            if i > 0 and i % args.progress_report_frequency == 0:
                logger.debug(f'{i}, {(time.time() - t)/i}')
    elif args.mode == 'logp':
        def split_fn_(l):
            a,b = l.split('\t')
            return a,b
        x_y = list(map(split_fn_, in_fp))
        x, y = zip(*x_y)
        for i,logp in enumerate(inference.gen_sents2logps(x, y, normalize=args.normalize)):
            if i % 1000 == 0:
                logger.debug(i)
            print(logp)

if __name__ == '__main__':
    main(sys.argv[1:], sys.stdin)
