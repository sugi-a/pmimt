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

from ...utils.beam_search import beam_search, create_length_penalty_fn
from ...vanilla import layers as vl
from ...language_model import layers as ll
from ...custom_text_data_pipeline import core as dp
from ...custom_text_data_pipeline.vocabulary import Vocabulary
from ..better_doc_level_trans_w_bayes.lattice_search import search_v2 as ltc_search

SF0 = tf.TensorSpec([], tf.float32)
SI0 = tf.TensorSpec([], tf.int32)
SI2 = tf.TensorSpec([None, None], tf.int32)
SB0 = tf.TensorSpec([], tf.bool)

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
        for f, t in model.id_substitutions_:
            y = tf.where(y == f, t, y)
        return model(y, training=False, cache=state, offsets=offsets)
    return f, state, shape_inv


def create_stateful_decoder_CtxLM(model, c, ntiles=None):
    if ntiles is not None:
        c = tf.repeat(c, ntiles, axis=0)
    B = tf.shape(c)[0]
    offsets = count_left_padding(c)
    f, state, shape_inv = create_stateful_decoder_LM(model, B, offsets=offsets)

    f(c) # Update state

    return f, state, shape_inv


def entropy_from_normalized_logp(logp):
    p = tf.math.exp(logp)
    return tf.math.reduce_sum(- p * logp, axis=-1)


def add_a_smoothing(logp, a):
    V = tf.cast(tf.shape(logp)[-1], tf.float32)
    p = tf.math.exp(logp)
    p += a / V
    p /= a + 1
    return tf.math.log(p)


def get_shallow_fusion_logits_from_decoders(tm, lm, dlm, y_in, beta):
    tm_logp = tf.nn.log_softmax(tm(y_in))
    dlm_logp = tf.nn.log_softmax(dlm(y_in))
    
    score = tm_logp + beta * dlm_logp

    return score


def get_fscore_logits_from_decoders(tm, lm, dlm, y_in, T1, T2, a, L1, topk=None, ret_pmi_pyc=False):
    tm_logp = tf.nn.log_softmax(tm(y_in))
    lm_logp = tf.nn.log_softmax(lm(y_in) / T2)
    dlm_logp = tf.nn.log_softmax(dlm(y_in) / T1)

    if a > 0:
        lm_logp = add_a_smoothing(lm_logp, a)
        dlm_logp = add_a_smoothing(dlm_logp, a)

    # [B, L, V]
    pmi = dlm_logp - lm_logp

    if topk is not None:
        # [B, L, k]
        tm_topk_v, tm_topk_i = tf.math.top_k(tm_logp, k=topk, sorted=True)

        # [B, L, 1]
        minimum = tm_topk_v[:, :, -1:]

        # Clipping {B, L, V}
        masked_idx = tm_logp < minimum
        pmi_clipped =tf.where(masked_idx, -12.0, pmi)
        tm_logp = tf.nn.log_softmax(tf.where(masked_idx, -12.0, tm_logp))

        # Standard deviation scaling
        # [B, L]
        std_tm_logp = tf.math.reduce_std(tm_topk_v, axis=-1)
        top_pmi = tf.gather(pmi_clipped, tm_topk_i, batch_dims=2)
        std_pmi = tf.math.reduce_std(top_pmi, axis=-1)
        target_std = tf.math.maximum(0.1, std_tm_logp - L1)
        scaler = tf.math.minimum(1.0, target_std / (std_pmi + 1e-6))

        # [B, L, V]
        pmi_clipped *= scaler[:,:,None]

        ### 
        #a = tf.nn.log_softmax(tm_topk_v)
        #b = tf.nn.log_softmax(tf.gather(pmi_clipped, tm_topk_i, batch_dims=2))

        #mask = create_mask(y_in)
        #std_tm = tf.math.reduce_std(a, axis=-1) * mask
        #std_pmi = tf.math.reduce_std(b, axis=-1) * mask
        #a = entropy_from_normalized_logp(a) * mask
        #b = entropy_from_normalized_logp(b) * mask
        #ntok = tf.reduce_sum(mask)
        #tf.print(
        #    tf.reduce_sum(tf.cast(a > b, tf.float32)),
        #    ntok,
        #    tf.reduce_sum(a) / ntok,
        #    tf.reduce_sum(b) / ntok,
        #    tf.reduce_sum(b-a) / ntok,
        #    tf.reduce_sum(std_tm) / ntok,
        #    tf.reduce_sum(std_pmi) / ntok,
        #    output_stream=sys.stderr)
        ###
    else:
        pmi_clipped = pmi

    fscore = tm_logp + pmi_clipped

    if ret_pmi_pyc:
        return fscore, pmi_clipped, dlm_logp
    else:
        return fscore


def get_tm_score_from_dec(tm, y):
    y_in, y_out = y[:, :-1], y[:, 1:]
    tm_logp = tf.nn.log_softmax(tm(y_in))
    res = tf.gather(tm_logp, y_out, batch_dims=2) * create_mask(y_out)
    return res

def get_lm_score_from_dec(lm, y, T):
    y_in, y_out = y[:, :-1], y[:, 1:]
    logp = tf.nn.log_softmax(lm(y_in) / T)
    res = tf.gather(logp, y_out, batch_dims=2) * create_mask(y_out)
    return res

def get_tok_fscore_from_decoders(tm, lm, dlm, y, T1, T2, a, L1, normalize):
    y_in, y_out = y[:, :-1], y[:, 1:]

    # [B, L, V]
    fscore_logits, pmi, pyc = get_fscore_logits_from_decoders(
        tm, lm, dlm, y_in, T1, T2, a, L1, ret_pmi_pyc=True)

    if normalize:
        fscore_logits = tf.nn.log_softmax(fscore_logits)

    # [B, L]
    gather_fn_ = lambda arg: tf.gather(arg, y_out, batch_dims=2) * create_mask(y_out)
    fscore, pmi, pyc = nest.map_structure(gather_fn_, (fscore_logits, pmi, pyc))

    return fscore, pmi, pyc


def get_tok_fscore(tm, lm, x, y, c, T1, T2, a, L1, normalize):
    B = tf.shape(x)[0]
    dec_tm, _, _ = create_stateful_decoder_TM(tm, x)
    dec_lm, _, _ = create_stateful_decoder_LM(lm, B)
    dec_clm, _, _ = create_stateful_decoder_CtxLM(lm, c)
    fscore, pmi, pyc = get_tok_fscore_from_decoders(
        dec_tm, dec_lm, dec_clm, y, T1, T2, a, L1, normalize)
    return fscore, pmi, pyc


def get_seq_fscore(tm, lm, x, y, c, T1, T2, a, L1, normalize):
    tok_fscore, tok_pmi, tok_pyc = get_tok_fscore(tm, lm, x, y, c, T1, T2, a, L1, normalize)
    reduce_fn_ = lambda arg: tf.reduce_sum(arg, axis=1)
    return nest.map_structure(reduce_fn_, (tok_fscore, tok_pmi, tok_pyc))


def ctx_aware_beam_search(
        tm, lm,
        x, c,
        beam_size,
        sos, eos,
        maxlen,
        normalize=False,
        T1=1.0, T2=1.0, lp=0.0, a=10, L1=1.0, shallow_fusion=False, beta=0.0):
    B = tf.shape(x)[0]
    K = beam_size
    sos = tf.broadcast_to(sos, [B])
    dec_tm, state_tm, sinv_tm = create_stateful_decoder_TM(tm, x, K)
    dec_lm, state_lm, sinv_lm = create_stateful_decoder_LM(lm, B, K)
    dec_clm, state_clm, sinv_clm = create_stateful_decoder_CtxLM(lm, c, K)
    
    state = [state_tm, state_lm, state_clm]
    shape_inv = [sinv_tm, sinv_lm, sinv_clm]

    def get_logits_fn_(y):
        if not shallow_fusion:
            fscores = get_fscore_logits_from_decoders(
                dec_tm, dec_lm, dec_clm, y, T1, T2, a, L1, topk=8)
            if normalize:
                fscores = tf.nn.log_softmax(fscores)

            return fscores
        else:
            sscores = get_shallow_fusion_logits_from_decoders(
                dec_tm, dec_lm, dec_clm, y, beta)
            if normalize:
                sscores = tf.nn.log_softmax(sscores)

            return sscores
    
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
        length_penalty_fn=create_length_penalty_fn(lp))
    
    return paths, scores
    

class Inference:
    def __init__(
            self, 
            transformer_model,
            decoder_language_model,
            vocab_source,
            vocab_target,
            batch_capacity):

        self.tm = transformer_model
        self.lm = decoder_language_model
        
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
        dtype = nest.map_structure(lambda x: tf.int32, structure)
        shape = nest.map_structure(lambda x: tf.TensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def create_dataset_multi(self, xs, vocabs):
        gen = self.create_data_gen_multi(zip(*xs), vocabs)
        return self.dataset_from_gen(gen, (None,) * len(vocabs))


    @tf.function(input_signature=[SI2, SI2, SI0, SF0, SB0, SF0, SF0])
    def comp_translate_shallow_fusion(self, x, c, beam_size, maxlen_ratio, normalize, lp, beta):
        with tf.device('/gpu:0'):
            if tf.size(x) > 0:
                src_lens = tf.reduce_sum(create_mask(x), axis=1)
                maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
                maxlen = tf.where(src_lens == 0.0, 0, maxlen)
                # [B, K, L], [B, K]

                paths, scores = ctx_aware_beam_search(
                    self.tm,
                    self.lm,
                    x,
                    c,
                    beam_size=beam_size,
                    sos=self.vocab_trg.SOS_ID,
                    eos=self.vocab_trg.EOS_ID,
                    maxlen=maxlen,
                    normalize=normalize,
                    lp=lp,
                    shallow_fusion=True,
                    beta=beta)
            else:
                B = tf.shape(x)[0]
                paths = tf.zeros([B, beam_size, 0], tf.int32)
                scores = tf.zeros([B, beam_size], tf.float32)

            return paths, scores


    @tf.function(input_signature=[SI2, SI2, SI0, SF0, SB0, SF0, SF0, SF0, SF0, SF0])
    def comp_translate(self, x, c, beam_size, maxlen_ratio, normalize, T1, T2, lp, a, L1):
        with tf.device('/gpu:0'):
            if tf.size(x) > 0:
                src_lens = tf.reduce_sum(create_mask(x), axis=1)
                maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
                maxlen = tf.where(src_lens == 0.0, 0, maxlen)
                # [B, K, L], [B, K]

                paths, scores = ctx_aware_beam_search(
                    self.tm,
                    self.lm,
                    x,
                    c,
                    beam_size=beam_size,
                    sos=self.vocab_trg.SOS_ID,
                    eos=self.vocab_trg.EOS_ID,
                    maxlen=maxlen,
                    normalize=normalize,
                    T1=T1,
                    T2=T2,
                    lp=lp,
                    a=a,
                    L1=L1)
            else:
                B = tf.shape(x)[0]
                paths = tf.zeros([B, beam_size, 0], tf.int32)
                scores = tf.zeros([B, beam_size], tf.float32)

            return paths, scores


    @tf.function(input_signature=[SI2, SI2, SI2, SF0, SF0, SF0, SF0, SB0])
    def comp_seq_logp(self, x, y, c, T1, T2, a, L1, normalize):
        with tf.device('/gpu:0'):
            fscore, _, _ = get_seq_fscore(self.tm, self.lm, x, y, c, T1, T2, a, L1, normalize)
            return fscore
    

    @tf.function(input_signature=[SI2, SI2, SF0, SF0, SF0, SF0, SB0])
    def comp_lm_scores(self, y, c, T1, T2, a, L1, normalize):
        with tf.device('/gpu:0'):
            x_ = tf.zeros([tf.shape(y)[0], 0], tf.int32)
            _, pmi, pyc = get_seq_fscore(self.tm, self.lm, x_, y, c, T1, T2, a, L1, normalize)
            return pmi, pyc

    

    @tf.function(input_signature=[SI2, SI0, SF0, SF0])
    def comp_fw_translate(self, x, beam_size, maxlen_ratio, lp):
        with tf.device('/gpu:0'):
            if tf.size(x) > 0:
                src_lens = tf.reduce_sum(create_mask(x), axis=1)
                maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
                maxlen = tf.where(src_lens == 0.0, 0, maxlen)
                # [B, K, L], [B, K]

                paths, scores = self.tm.beam_search_decode_with_prefix(
                    x,
                    prefix_or_sos=self.vocab_trg.SOS_ID,
                    eos=self.vocab_trg.EOS_ID,
                    beam_size=beam_size,
                    maxlen=maxlen,
                    length_penalty_fn=create_length_penalty_fn(lp))
            else:
                B = tf.shape(x)[0]
                paths = tf.zeros([B, beam_size, 0], tf.int32)
                scores = tf.zeros([B, beam_size], tf.float32)

            return paths, scores
    

    @tf.function(input_signature=[SI2])
    def comp_lm_logp(self, y):
        y_in, y_out = y[:, :-1], y[:, 1:]
        
        logits = self.lm(y_in, training=False)
        logp_distr = tf.nn.log_softmax(logits)

        logp = tf.gather(logp_distr, y_out, batch_dims=2) * create_mask(y_out)
        return tf.reduce_sum(logp, axis=1)


    @tf.function(input_signature=[SI2, SI2])
    def comp_cond_lm_score(self, ctx, y):
        ctx, offsets = vl.transfer_padding_to_left(ctx)
        y_in, y_out = y[:, :-1], y[:, 1:]
        
        joint_input = tf.concat([ctx, y_in], axis=1)
        logits = self.lm(joint_input, training=False, offsets=offsets)
        logits = logits[:, -tf.shape(y_out)[1]:]
        logp_distr = tf.nn.log_softmax(logits)

        logp = tf.gather(logp_distr, y_out, batch_dims=2) * create_mask(y_out)
        return tf.reduce_sum(logp, axis=1)


    def translate_docs_beam_shortcut(self, docs, n_ctx, beam_size, maxlen_ratio=1.5, normalize=False, T1=1.0, T2=1.0, lp=0.0, a=0, L1=1.0):
        src_v, trg_v = self.vocab_src, self.vocab_trg

        def make_batches2_(sents_list, vocabs):
            o = zip(*sents_list)
            o = dp.gen_line2IDs_multi(o, vocabs)
            o = dp.gen_batch_of_capacity_multi(o, self.batch_capacity)
            o = dp.gen_pad_batch_multi(o)
            o = dp.gen_list2numpy_nested(o)
            return o

        # Serialize
        sources = []
        for doc in docs: sources.extend(doc)

        # Sentence-level Translation
        logger.debug('Start sentence-level translation')
        t = time.time()

        trans = []
        for x, in make_batches2_((sources,), (src_v,)):
            # [Batch, Beam, L]
            hypos, _ = self.comp_fw_translate(x, beam_size, maxlen_ratio, lp)
            hypos = hypos[:, 0].numpy()
            hypos = trg_v.IDs2text(hypos)
            hypos = [trg_v.ID2tok[trg_v.EOS_ID]  + ' ' + line for line in hypos]
            trans.extend(hypos)
            
            logger.debug(
                f'{len(trans)} / {len(sources)}, {(time.time() - t)/len(trans)}')

        assert len(trans) == len(sources)

        # Deserialize
        trans_docs = []
        i = 0
        for doc in docs:
            trans_docs.append(trans[i: i + len(doc)])
            i += len(doc)

        # Create Context (serialized)
        ctxs = []
        for t_doc in trans_docs:
            for i in range(len(t_doc)):
                ctxs.append(' '.join(t_doc[max(0, i - 3): i]))

        assert len(ctxs) == len(sources)

        # Context aware translation
        logger.debug('Start context-aware translation')
        t = time.time()
        out = []
        for x, c in make_batches2_((sources, ctxs), (src_v, trg_v)):
            hypos, _ = self.comp_translate(
                x, c, beam_size, maxlen_ratio, normalize,
                T1, T2, lp, a, L1)
            hypos = hypos[:, 0].numpy()
            hypos = trg_v.IDs2text(hypos)
            out.extend(hypos)

            logger.debug(f'{len(out)} / {len(sources)}, {(time.time()-t)/len(out)}')

        # Deserialize
        out_docs = []
        i = 0
        for doc in docs:
            out_docs.append(out[i: i + len(doc)])
            i += len(doc)
        assert len(out_docs) == len(docs)
        return out_docs


    def translate_doc_rerank(
            self,
            doc, n_ctx,
            lattice_width, lattice_beam_size,
            maxlen_ratio=1.5,
            normalize=False,
            T1=1.0, T2=1.0, lp=0.0, a=10):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        LW = lattice_width
        LD = len(doc)
        LB = lattice_beam_size

        def batch_and_pad_(list_of_seqs):
            o = zip(*list_of_seqs)
            o = dp.gen_batch_of_capacity_multi(o, self.batch_capacity)
            o = dp.gen_pad_batch_multi(o)
            o = dp.gen_list2numpy_nested(o)
            return o

        def make_batches_(sents_list, vocabs):
            o = zip(*sents_list)
            o = dp.gen_line2IDs_multi(o, vocabs)
            o = dp.gen_batch_of_capacity_multi(o, self.batch_capacity)
            o = dp.gen_pad_batch_multi(o)
            o = dp.gen_list2numpy_nested(o)
            return o

        # Generate hypotheses and forward scores with the forward model
        hypos = []
        fw_scores = []
        for x, in make_batches_((doc,), (src_v)):
            h_, s_ = self.comp_fw_translate(
                x, lattice_width, maxlen_ratio, lp)
            for b in h_.numpy():
                b = [
                    [trg_v.EOS_ID] + [_id for _id in seq if _id not in trg_v.ctrls]
                    for seq in b
                ]
                #b = trg_v.drop_ctrls(b)
                #b = [[trg_v.EOS_ID] + seq for seq in b]
                hypos.append(b)
            fw_scores.extend(s_.numpy().tolist())

        # <LD, LM>
        fw_scores = np.array(fw_scores)

        # Compute LM scores
        hypos_flat = list(chain(*hypos))

        lm_scores = []

        for y, in batch_and_pad_((hypos_flat,)):
            lm_scores.extend(self.comp_lm_logp(y).numpy())

        lm_scores = np.array(lm_scores).reshape(LD, LW)

        def get_score_at(pos, paths):
            # DLM score
            ctx_start = max(0, pos - n_ctx)
            ctx = [
                list(chain.from_iterable(
                    hypos[n][paths[b][n]]
                    for n in range(ctx_start, pos)
                )) for b in range(LB)
            ]
            ctx_tiled = list(chain.from_iterable((seq,)*LW for seq in ctx))

            cur_hypos = hypos[pos]
            hypos_tiled = cur_hypos * LB

            dataset = batch_and_pad_((ctx_tiled, hypos_tiled))
            dlm_scores = []
            for c, y in dataset:
                dlm_scores.extend(self.comp_cond_lm_score(c, y).numpy())

            dlm_score_ = np.array(dlm_scores)
            lm_score_ = np.tile(lm_scores[pos], LB)
            fw_score_ = np.tile(fw_scores[pos], LB)

            score = (
                fw_score_
                + dlm_score_ / T1
                - lm_score_ / T2)
            
            return score


        idx = ltc_search(LW, LD, LB, get_score_at)

        out = [hypos[i][idx[i]] for i in range(LD)]
        out = trg_v.IDs2text(out)
        return out


    def translate_docs(
            self, docs, n_ctx, beam_size, maxlen_ratio=1.5):
        return [self.translate_doc(doc, n_ctx, beam_size, maxlen_ratio)
            for doc in docs]


    def translate_docs_batch(self, docs, n_ctx, beam_size, maxlen_ratio=1.5, normalize=False, T1=1.0, T2=1.0, lp=0.0, a=0, L1=1.0, shallow_fusion=False, beta=0.0):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        def make_batches_(src_sents, ctx_sents):
            o = dp.gen_line2IDs_multi(
                zip(src_sents, ctx_sents),
                (src_v, trg_v))
            o = dp.gen_batch_of_capacity_multi(o, self.batch_capacity / beam_size)
            o = dp.gen_pad_batch_multi(o)
            o = dp.gen_list2numpy_nested(o)
            return o

        docs = [deque(doc) for doc in docs]
        out = [[] for i in range(len(docs))]
        ctx_q = [deque() for i in range(len(docs))]
        len_q = [deque() for i in range(len(docs))]

        t = time.time()
        ntotal = sum(map(len, docs))
        nprocessed = 0

        while True:
            idx = [i for i, doc in enumerate(docs) if len(doc) > 0]

            if len(idx) == 0:
                break

            src_sents = [docs[i].popleft() for i in idx]
            ctx_sents = [' '.join(ctx_q[i]) for i in idx]

            x_cs = make_batches_(src_sents, ctx_sents)
            os = []
            for x, c in x_cs:
                if not shallow_fusion:
                    paths, _ = self.comp_translate(x, c, beam_size, maxlen_ratio, normalize, T1, T2, lp, a, L1)
                else:
                    paths, _ = self.comp_translate_shallow_fusion(x, c, beam_size, maxlen_ratio, normalize, lp, beta)
                paths = paths[:, 0].numpy()
                trans_toks = trg_v.IDs2tokens2D(paths)
                os.extend(trans_toks)

            assert len(os) == len(idx)
            for i, o in zip(idx, os):
                out[i].append(' '.join(o))
                len_q[i].append(1 + len(o))
                ctx_q[i].append(trg_v.ID2tok[trg_v.EOS_ID])
                ctx_q[i].extend(o)
                
                if len(len_q[i]) > n_ctx:
                    l = len_q[i].popleft()
                    for _ in range(l):
                        ctx_q[i].popleft()

            nprocessed += len(idx)
            logger.debug(f'{nprocessed}/{ntotal}\t{(time.time()-t)/nprocessed}')

        return out

    def gen_fscore(self, x, y, c, normalize_categorically, T1=1.0, T2=1.0, a=0, L1=1.0):
        """
            y: <s> EEE FFF </s>
            c: </s> AAA BBB </s> CCC DDD
        """
        dataset = (
            self.create_dataset_multi(
                (x, y, c),
                (self.vocab_src, self.vocab_trg, self.vocab_trg))
            .prefetch(1)
        )

        for x, y, c in dataset:
            c, _ = vl.transfer_padding_to_left(c)
            scores = self.comp_seq_logp(x, y, c, T1, T2, a, L1, normalize_categorically)
            for score in scores.numpy():
                yield score


    def gen_pmi_pyc(self, y, c, normalize_categorically, T1=1.0, T2=1.0, a=0, L1=-1000):
        """
            y: <s> EEE FFF </s>
            c: </s> AAA BBB </s> CCC DDD
        """
        dataset = (
            self.create_dataset_multi(
                (y, c),
                (self.vocab_trg, self.vocab_trg))
            .prefetch(1)
        )

        for y, c in dataset:
            c, _ = vl.transfer_padding_to_left(c)
            pmi, pyc = self.comp_lm_scores(y, c, T1, T2, a, L1, normalize_categorically)
            for pmi_, pyc_ in zip(pmi.numpy(), pyc.numpy()):
                yield pmi_, pyc_


    def unit_test(self):
        t = time.time()
        with open('./inf_test/dev.ru') as f:
            for i, line in enumerate(self.gen_sents2sents( f, beam_size=1)):
                a = line
                if i % 100 == 0 and i != 0:
                    print(i, (time.time() - t)/i)


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
    p.add_argument('tm_dir', type=str)
    p.add_argument('lm_dir', type=str)
    p.add_argument('--tm_checkpoint', '--tm_ckpt', type=str)
    p.add_argument('--lm_checkpoint', '--lm_ckpt', type=str)
    p.add_argument('--capacity', type=int, default=8000)
    p.add_argument('--mode',
        choices=[
            'translate',
            'trans_shallow_fusion',
            'rerank',
            'trans_beam_shortcut',
            'fscore',
            'instance',
        ],
        default='translate')
    p.add_argument('--beam_size', type=int, default=1)
    p.add_argument('--n_ctx', type=int, default=3)
    p.add_argument('--length_penalty', type=float, default=0.0)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--debug_eager_function', action='store_true')
    p.add_argument('--progress_frequency', type=int, default=10**10)
    p.add_argument('-T', type=float)
    p.add_argument('-T1', type=float, default=1.0)
    p.add_argument('-T2', type=float, default=1.0)
    p.add_argument('-a', type=float, default=0.0)
    p.add_argument('-L1', type=float, default=-1000)
    p.add_argument('--beta', type=float, default=0.0) # for shallow fusion
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--maxlen_ratio', type=float, default=2.0)
    p.add_argument('--lattice_width', type=int)
    p.add_argument('--lattice_beam_size', type=int)

    args = p.parse_args(argv)

    if args.debug:
        basicConfig(level=DEBUG)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    tm = load_tm(args.tm_dir, args.tm_checkpoint)
    lm = load_lm(args.lm_dir, args.lm_checkpoint)

    with open(f'{args.tm_dir}/vocab_config.json') as f:
        vc = json.load(f)

    vocab_src = Vocabulary(
        args.tm_dir + '/' + vc['source_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    vocab_trg = Vocabulary(
        args.tm_dir + '/' + vc['target_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])

    # Shortcut
    if args.T is not None:
        args.T1 = args.T
        args.T2 = args.T
        logger.info(f'Overwrite T1 and T2 by {args.T}')

    # Inference Class
    inference = Inference(tm, lm, vocab_src, vocab_trg, args.capacity)

    if args.mode == 'translate':
        docs = gen_lines_to_docs(in_fp)
        out_docs = inference.translate_docs_batch(
            docs,
            n_ctx=args.n_ctx,
            beam_size=args.beam_size,
            maxlen_ratio=2.0,
            normalize=args.normalize,
            T1=args.T1,
            T2=args.T2,
            lp=args.length_penalty,
            a=args.a,
            L1=args.L1)
        for line in gen_docs_to_lines(out_docs):
            print(line)
    elif args.mode == 'trans_shallow_fusion':
        docs = gen_lines_to_docs(in_fp)
        out_docs = inference.translate_docs_batch(
            docs,
            n_ctx=args.n_ctx,
            beam_size=args.beam_size,
            maxlen_ratio=2.0,
            normalize=args.normalize,
            lp=args.length_penalty,
            shallow_fusion=True,
            beta=args.beta)
        for line in gen_docs_to_lines(out_docs):
            print(line)

    elif args.mode == 'fscore':
        def split_(l):
            a,b,c = l.split('\t')
            return a,b,c
        in_fp = list(in_fp)
        x_y_c = list(map(split_, in_fp))
        #x, y, c = dp.fork_iterable(x_y_c, 3)
        x, y, c = zip(*x_y_c)
        for line in inference.gen_fscore(x, y, c, args.normalize, args.T1, args.T2, args.a, args.L1):
            print(line)
    elif args.mode == 'rerank':
        docs = list(gen_lines_to_docs(in_fp))
        total = sum(map(len, docs))
        processed = 0
        out_docs = []
        t = time.time()
        for doc in docs:
            odoc = inference.translate_doc_rerank(
                doc,
                args.n_ctx,
                args.lattice_width,
                args.lattice_beam_size,
                args.maxlen_ratio,
                args.normalize,
                args.T1, args.T2, args.length_penalty)
            out_docs.append(odoc)
            processed += len(doc)
            logger.debug(
                f'{processed} / {total}, {(time.time()-t)/processed}')

        for line in gen_docs_to_lines(out_docs):
            print(line)
    elif args.mode == 'trans_beam_shortcut':
        docs = list(gen_lines_to_docs(in_fp))
        total = sum(map(len, docs))
        processed = 0
        out_docs = inference.translate_docs_beam_shortcut(
            docs,
            args.n_ctx,
            args.beam_size,
            args.maxlen_ratio,
            args.normalize,
            args.T1, args.T2,
            args.length_penalty,
            a=args.a,
            L1=args.L1)
        for line in gen_docs_to_lines(out_docs):
            print(line)


if __name__ == '__main__':
    main(sys.argv[1:], sys.stdin)
