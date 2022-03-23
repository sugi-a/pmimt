from logging import getLogger, DEBUG, basicConfig; logger = getLogger(__name__)
import argparse
import sys
import json
import time
from collections import deque
from itertools import chain

import tensorflow as tf
from tensorflow import keras, nest
import numpy as np

from ...utils.beam_search import beam_search
from ...vanilla import layers as vl
from ...language_model import layers as ll
from ...custom_text_data_pipeline import core as dp
from ...custom_text_data_pipeline.vocabulary import Vocabulary

from .lattice_search import search as ltc_search

SF0 = tf.TensorSpec([], tf.float32)
SI0 = tf.TensorSpec([], tf.int32)
SI2 = tf.TensorSpec([None, None], tf.int32)


def gen_lines_to_docs(line_iterable):
    doc = []
    for line in line_iterable:
        line = line.strip()
        if len(line) == 0:
            if len(doc) > 0:
                yield doc
                doc = []
        else:
            doc.append(line)
            
    if len(doc) > 0:
        yield doc


def gen_docs_to_lines(docs_iterable):
    for i, doc in enumerate(docs_iterable):
        if i > 0:
            yield ''
        yield from doc


def recursive_update(target, ref):
    """Recursively update the nested structure `target` by `ref`"""
    containers = (list, dict) # tuple is not allowed
    assert isinstance(ref, containers)
    
    if isinstance(ref, list):
        assert len(ref) == len(target)
        for i in range(len(ref)):
            if isinstance(ref[i], containers):
                recursive_update(target[i], ref[i])
            else:
                target[i] = ref[i]
    else:
        assert target.keys() == ref.keys()
        for k in ref.keys():
            if isinstance(ref[k], containers):
                recursive_update(target[k], ref[k])
            else:
                target[k] = ref[k]


def create_mask(seq, dtype=tf.float32):
    return tf.cast(seq != 0, dtype)


def count_left_padding(seq, dtype=tf.int32):
    pads = tf.cast(seq == 0, dtype)
    left_pads = tf.math.cumprod(pads, axis=1)
    return tf.reduce_sum(left_pads, axis=1)


def create_stateful_decoder_TM(model, x, ntiles=None):
    enc_out, enc_pad_bias = model.encode(x, training=False)

    if ntiles is not None:
        enc_out, enc_pad_bias = nest.map_structure(
            lambda x: tf.repeat(x, ntiles, axis=0),
            (enc_out, enc_pad_bias))

    state, shape_inv = model.create_cache(tf.shape(enc_out)[0])
    return (
        model.create_decoder(enc_out, enc_pad_bias, state),
        state,
        shape_inv
    )


def create_stateful_decoder_LM(model, B, ntiles=None, offsets=None):
    if ntiles is not None:
        B *= ntiles
    state, shape_inv = model.create_cache(B)
    def f(y):
        if hasattr(model, 'id_substitutions_'):
            for f, t in model.id_substitutions_:
                y = tf.where(y == f, t, y)
        return model(y, training=False, cache=state, offsets=None)
    return f, state, shape_inv


def create_stateful_decoder_CtxLM(model, c, ntiles=None):
    if ntiles is not None:
        c = tf.repeat(c, ntiles, axis=0)
    B = tf.shape(c)[0]
    offsets = count_left_padding(c)
    f, state, shape_inv = create_stateful_decoder_LM(model, B, offsets=offsets)

    f(c) # Update state

    return f, state, shape_inv


def get_fscore_logits_from_decoders(tm, lm, dlm, y_in):
    tm_logp = tf.nn.log_softmax(tm(y_in))
    lm_logp = tf.nn.log_softmax(lm(y_in))
    dlm_logp = tf.nn.log_softmax(dlm(y_in))

    return tm_logp + dlm_logp - lm_logp


def get_tok_fscore_from_decoders(tm, lm, dlm, y):
    y_in, y_out = y[:, :-1], y[:, 1:]

    # [B, L, V]
    fscore_logits = get_fscore_logits_from_decoders(tm, lm, dlm, y_in)

    # [B, L]
    return tf.gather(fscore, y_out) * create_mask(y_out)


def get_tok_fscore(tm, lm, x, y, c):
    B = tf.shape(x)[0]
    dec_tm, _, _ = create_stateful_decoder_TM(tm, x)
    dec_lm, _, _ = create_stateful_decoder_LM(lm, B)
    dec_clm, _, _ = create_stateful_decoder_CtxLM(lm, c)
    return get_tok_fscore_from_decoders(dec_tm, dec_lm, dec_clm, y)


def get_seq_fscore(tm, lm, x, y, c):
    tok_fscore = get_tok_fscore(tm, lm, x, y, c)
    return tf.reduce_sum(tok_fscore, axis=1)


def ctx_aware_beam_search(tm, lm, x, c, beam_size, sos, eos, maxlen):
    B = tf.shape(x)[0]
    K = beam_size
    sos = tf.broadcast_to(sos, [B])
    dec_tm, state_tm, sinv_tm = create_stateful_decoder_TM(tm, x, K)
    dec_lm, state_lm, sinv_lm = create_stateful_decoder_LM(lm, B, K)
    dec_clm, state_clm, sinv_clm = create_stateful_decoder_CtxLM(lm, c, K)
    
    state = [state_tm, state_lm, state_clm]
    shape_inv = [sinv_tm, sinv_lm, sinv_clm]

    def get_logits_fn_(y):
        return get_fscore_logits_from_decoders(
            dec_tm,
            dec_lm,
            dec_clm,
            y)
    
    def perm_batch_fn_(permutation):
        tm.permute_cache(state_tm, permutation)
        lm.permute_cache(state_lm, permutation)
        lm.permute_cache(state_clm, permutation)
    
    def get_state_fn_():
        return state
    
    def put_controlled_state_fn_(state_):
        recursive_update(state, state_)
    
    paths, scores = beam_search(
        get_logits_fn=get_logits_fn_,
        perm_batch_fn=perm_batch_fn_,
        sos=sos,
        eos=eos,
        beam_size=beam_size,
        maxlen=maxlen,
        pad=0,
        get_state_fn=get_state_fn_,
        put_controlled_state_fn=put_controlled_state_fn_,
        shape_invariants=shape_inv,
        length_penalty_fn=None)
    
    return paths, scores
    

class Inference:
    def __init__(
            self, 
            fw_tm,
            bw_tm,
            lm,
            vocab_source,
            vocab_target,
            batch_capacity):

        self.fw_tm = fw_tm
        self.bw_tm = bw_tm
        self.lm = lm
        
        self.vocab_src = vocab_source
        self.vocab_trg = vocab_target
        
        self.batch_capacity = batch_capacity

        # ID substitution for the language model
        # Must be improved
        self.lm.id_substitutions_ = [
            (self.vocab_trg.SOS_ID, self.vocab_trg.EOS_ID)
        ]


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


    @tf.function(input_signature=[SI2, SI0, SF0])
    def comp_fw_translate(self, x, beam_size, maxlen_ratio):
        with tf.device('/gpu:0'):
            if tf.size(x) > 0:
                src_lens = tf.reduce_sum(create_mask(x), axis=1)
                maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
                maxlen = tf.where(src_lens == 0.0, 0, maxlen)
                # [B, K, L], [B, K]

                paths, scores = self.fw_tm.beam_search_decode_with_prefix(
                    x,
                    prefix_or_sos=self.vocab_trg.SOS_ID,
                    eos=self.vocab_trg.EOS_ID,
                    beam_size=beam_size,
                    maxlen=maxlen)
            else:
                B = tf.shape(x)[0]
                paths = tf.zeros([B, beam_size, 0], tf.int32)
                scores = tf.zeros([B, beam_size], tf.float32)

            return paths, scores
    

    @tf.function(input_signature=[SI2, SI2])
    def comp_fw_logp(self, x, y):
        y_in, y_out = y[:, :-1], y[:, 1:]
        logits = self.fw_tm(x, y_in, training=False)
        logp_dist = tf.nn.log_softmax(logits)

        logp = tf.gather(logp_dist, y_out, batch_dims=2) * create_mask(y_out)

        return tf.reduce_sum(logp, axis=1)


    @tf.function(input_signature=[SI2, SI2])
    def comp_bw_logp(self, x, y):
        x_in, x_out = x[:, :-1], x[:, 1:]
        logits = self.bw_tm(y, x_in, training=False)
        logp_dist = tf.nn.log_softmax(logits)

        logp = tf.gather(logp_dist, x_out, batch_dims=2) * create_mask(x_out)

        return tf.reduce_sum(logp, axis=1)
    

    @tf.function(input_signature=[SI2, SI2])
    def comp_cond_LM(self, ctx, y):
        # SOS substitution
        y = tf.where(y == self.vocab_trg.SOS_ID, self.vocab_trg.EOS_ID, y)
        ctx = tf.where(ctx == self.vocab_trg.SOS_ID, self.vocab_trg.EOS_ID, ctx)

        ctx, offsets = vl.transfer_padding_to_left(ctx)
        y_in, y_out = y[:, :-1], y[:, 1:]
        
        joint_input = tf.concat([ctx, y_in], axis=1)
        logits = self.lm(joint_input, training=False, offsets=offsets)
        logits = logits[:, -tf.shape(y_out)[1]:]
        logp_distr = tf.nn.log_softmax(logits)

        logp = tf.gather(logp_distr, y_out, batch_dims=2) * create_mask(y_out)
        return tf.reduce_sum(logp, axis=1)


    @tf.function(input_signature=[SI2, SI2, SI2, SF0, SF0, SF0])
    def comp_rerank_score(self, x, y, c, l1, l2, l3):
        """c must not include the separator symbol at the end"""
        # [B]
        fw_score = self.comp_fw_logp(x, y)
        bw_score = self.comp_bw_logp(x, y)
        lm_score = self.comp_cond_LM(c, y)
        lens = tf.math.reduce_sum(create_mask(y[:, 1:]), axis=1)
        return (
            lm_score
            + l1 * fw_score
            + l2 * bw_score
            + l3 * lens)


    def translate_doc(
            self,
            doc, n_ctx,
            lattice_width, lattice_beam_size,
            l1, l2, l3, maxlen_ratio):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        LW = lattice_width
        D = len(doc)

        def wrap_(lines, vocab, sos=None, eos=None):
            o = lines
            SOS = vocab.ID2tok[vocab.SOS_ID] if sos is None else sos
            EOS = vocab.ID2tok[vocab.EOS_ID] if eos is None else eos
            if sos is not False:
                o = map(lambda x: f'{SOS} {x}', o)
            if eos is not False:
                o = map(lambda x: f'{x} {EOS}', o)
            return o

        def make_batches_(sents_list, vocabs):
            o = zip(*sents_list)
            o = dp.gen_line2IDs_multi(o, vocabs)
            o = dp.gen_batch_of_capacity_multi(o, self.batch_capacity)
            o = dp.gen_pad_batch_multi(o)
            o = dp.gen_list2numpy_nested(o)
            return o

        # Generate hypos and forward scores with the forward model
        # list<list<naked_sents, LW>, D>
        hypos = []
        fw_scores = []
        for batch, in make_batches_((doc,), (src_v,)):
            h_, s_ = self.comp_fw_translate(batch, lattice_width, maxlen_ratio)
            for b in h_.numpy():
                # b: <[LW, L]>
                hypos.append(trg_v.IDs2text(b, skip_control_symbols=True))
            fw_scores.extend(s_.numpy().tolist())
        
        # <D, LW>
        fw_scores = np.array(fw_scores)

        # Compute backward scores
        bw_scores = []
        # [D * LW]
        hypos_flat = list(chain(*hypos))
        hypos_flat = wrap_(hypos_flat, trg_v)
        src_tiled = list(chain.from_iterable((sent,)*LW for sent in doc))
        for x, y in make_batches_((src_tiled, hypos_flat), (src_v, trg_v)):
            bw_scores.extend(self.comp_bw_logp(x, y))

        # <D, LW>
        bw_scores = np.array(bw_scores).reshape([D, LW])

        # Beam search on lattice. [D]
        EOS = trg_v.ID2tok[trg_v.EOS_ID]
        hypo_ids = [list(dp.gen_line2IDs(wrap_(h_, None, sos=EOS, eos=False), trg_v)) for h_ in hypos]
        idx = ltc_search(
            hypo_ids,
            fw_scores, bw_scores,
            self.comp_cond_LM,
            l1, l2, l3,
            lattice_beam_size, n_ctx)

        out = [hypos[i][idx[i]] for i in range(D)]
        return out


    def translate_docs(
            self,
            docs, n_ctx,
            lattice_width, lattice_beam_size,
            l1, l2, l3,
            maxlen_ratio=2.0):
        res = []
        t = time.time()
        docs = list(docs)
        total = sum(map(len, docs))
        n = 0
        for doc in docs:
            res.append(
                self.translate_doc(
                    doc,
                    n_ctx,
                    lattice_width,
                    lattice_beam_size,
                    l1, l2, l3,
                    maxlen_ratio)
                )
            n += len(doc)
            logger.debug(f'{n}/{total}\t{(time.time() - t)/n}')
        return res


    def gen_reranking_score(self, tab_sep_x_c_y_lines, l1, l2, l3):
        def split_fn_(l):
            a,b,c = l.split('\t')
            return a,b,c

        data_gen = self.create_data_gen_multi(
            map(split_fn_, tab_sep_x_c_y_lines),
            (self.vocab_src, self.vocab_trg, self.vocab_trg))
        
        for x, c, y in dp.gen_list2numpy_nested(data_gen()):
            for s in self.comp_rerank_score(
                    x, c=c, y=y, l1=l1, l2=l2, l3=l3).numpy():
                yield s


def load_tm(tm_dir, checkpoint):
    # Translation model Config
    with open(f'{tm_dir}/model_config.json') as f:
        model_config = json.load(f)
    
    # Transformer Model
    model = vl.Transformer.from_config(model_config)
    ckpt = tf.train.Checkpoint(model=model)
    if checkpoint is None:
        ckpt_path = tf.train.latest_checkpoint(f'{tm_dir}/checkpoint_best')
    else:
        ckpt_path = checkpoint
    assert ckpt_path is not None
    ckpt.restore(ckpt_path)
    logger.info(f'Checkpoint: {ckpt_path}')

    return model


def load_lm(lm_dir, checkpoint):
    # Translation model Config
    with open(f'{lm_dir}/model_config.json') as f:
        model_config = json.load(f)
    
    # Transformer Model
    model = ll.DecoderLanguageModel.from_config(model_config)
    ckpt = tf.train.Checkpoint(model=model)
    if checkpoint is None:
        ckpt_path = tf.train.latest_checkpoint(f'{lm_dir}/checkpoint_best')
    else:
        ckpt_path = checkpoint
    assert ckpt_path is not None
    ckpt.restore(ckpt_path)
    logger.info(f'Checkpoint: {ckpt_path}')

    return model


def main(argv, in_fp):
    p = argparse.ArgumentParser()
    p.add_argument('fw_tm_dir', type=str)
    p.add_argument('bw_tm_dir', type=str)
    p.add_argument('lm_dir', type=str)
    p.add_argument('l1', type=float)
    p.add_argument('l2', type=float)
    p.add_argument('l3', type=float)
    p.add_argument('--fw_tm_checkpoint', '--fw_ckpt', type=str)
    p.add_argument('--bw_tm_checkpoint', '--bw_ckpt', type=str)
    p.add_argument('--lm_checkpoint', '--lm_ckpt', type=str)
    p.add_argument('--capacity', type=int, default=16384)
    p.add_argument('--mode', choices=['translate', 'rerank_score'], default='translate')
    p.add_argument('--lattice_width', type=int, default=20)
    p.add_argument('--lattice_beam_size', type=int, default=5)
    p.add_argument('--n_ctx', type=int, default=3)
    p.add_argument('--length_penalty', type=float)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--debug_eager_function', action='store_true')
    p.add_argument('--progress_frequency', type=int, default=10**10)
    args = p.parse_args(argv)

    if args.debug:
        basicConfig(level=DEBUG)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    fw_tm = load_tm(args.fw_tm_dir, args.fw_tm_checkpoint)
    bw_tm = load_tm(args.bw_tm_dir, args.bw_tm_checkpoint)
    lm = load_lm(args.lm_dir, args.lm_checkpoint)

    with open(f'{args.fw_tm_dir}/vocab_config.json') as f:
        vc = json.load(f)

    vocab_src = Vocabulary(
        args.fw_tm_dir + '/' + vc['source_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    vocab_trg = Vocabulary(
        args.fw_tm_dir + '/' + vc['target_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    
    # Inference Class
    inference = Inference(
        fw_tm, bw_tm, lm, vocab_src, vocab_trg, args.capacity)

    if args.mode == 'translate':
        docs = gen_lines_to_docs(in_fp)
        out_docs = inference.translate_docs(
            docs,
            n_ctx=args.n_ctx,
            lattice_width=args.lattice_width,
            lattice_beam_size=args.lattice_beam_size,
            l1=args.l1,
            l2=args.l2,
            l3=args.l3,
            maxlen_ratio=2.0)
        for line in gen_docs_to_lines(out_docs):
            print(line)
    elif args.mode == 'rerank_score':
        for s in inference.gen_reranking_score(
                in_fp, args.l1, args.l2, args.l3):
            print(s)


if __name__ == '__main__':
    main(sys.argv[1:], sys.stdin)
