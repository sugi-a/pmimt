from logging import getLogger; logger = getLogger(__name__)
from logging import basicConfig, ERROR, WARNING, INFO, DEBUG, NOTSET
import sys
import random
import argparse
import json
import time
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.nest import map_structure
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from ...custom_text_data_pipeline.vocabulary import Vocabulary
from ...custom_text_data_pipeline import core as dp
from ...utils import multi_gpu as mg
from ...vanilla import train as vt
from .layers import DocTransformer


def split_src_ctx_and_main(src_trg, omit_ctx=False):
    src, trg = src_trg
    ctx, main = src.split('\t')
    if omit_ctx:
        return "", main, trg
    else:
        return ctx, main, trg

__DEBUG_printed_trainable_vars__ = False

class Train(vt.Train):
    def __init__(self, pretrain, *args, gpus=None, accums=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.gpus = vt.get_visible_gpus() if gpus is None else gpus
        logger.info(f'#GPUs {self.gpus}')

        self.pretrain = pretrain
        logger.info(f'Pretraining: {pretrain}')
        
        self.accums = 1 if accums is None else accums

        # Debug
        N2 = tf.TensorSpec([None, None], tf.int32)
        self.model_call_wrapper = tf.function(
            lambda c,x,y,training: self.model(c, x, y, training=training),
            input_signature=[N2, N2, N2, tf.TensorSpec([], tf.bool)])


    def get_batch_weight_core(self, batch):
        c, x, y = batch
        return vt.count_toks(y[:, 1:], tf.float32)


    def get_batch_weight(self, batch):
        ys = nest.flatten([[b[2] for b in row] for row in batch])
        return tf.math.add_n([vt.count_toks(y[:, 1:]) for y in ys])


    def calc_grad_metrics_core(self, batch):
        c, x, y = batch
        y_i, y_o = y[:, :-1], y[:, 1:]
        ls_eps = self.train_config['label_smoothing']
        with tf.GradientTape() as tape:
            #logits = self.model(c, x, y_i, training=True)
            logits = self.model_call_wrapper(c, x, y_i, True)
            loss = vt.loss_norm(logits, y_o, ls_eps=ls_eps)

        global __DEBUG_printed_trainable_vars__
        if not __DEBUG_printed_trainable_vars__:
            __DEBUG_printed_trainable_vars__ = True
            logger.debug('\n'.join(map(str, self.model.trainable_variables)))
        grad = tape.gradient(
            loss,
            self.model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        metrics = {k: v['calc'](logits, y_o, loss)
            for k, v in self.metrics.items()}

        return grad, metrics


    def calc_grad_metrics(self, inputs):
        core_fn = self.calc_grad_metrics_core
        count_fn = self.get_batch_weight_core

        fn = lambda b: (core_fn(b), count_fn(b))
        o_specs = vt.get_output_specs_shape_inv(fn, inputs[0][0])

        def accum_fn(batches):
            g_ms, ntoks = zip(*mg.sequential_map(fn, batches, o_specs))
            return vt.weighted_avg(g_ms, ntoks)

        g_ms, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        g_ms, _ = vt.weighted_avg(g_ms, ntoks)

        return g_ms

    
    def calc_metrics_core(self, batch):
        c, x, y = batch
        if tf.size(x) > 0:
            y_i, y_o = y[:, :-1], y[:, 1:]
            #logits = self.model(c, x, y_i, training=False)
            logits = self.model_call_wrapper(c, x, y_i, False)
            loss = vt.loss_norm(
                logits, y_o, ls_eps=self.train_config['label_smoothing'])
            metrics = {k: v['calc'](logits, y_o, loss)
                for k, v in self.metrics.items()}
        else:
            metrics = {k: 0.0 for k in self.metrics.keys()}

        return metrics


    def calc_metrics(self, inputs):
        core_fn = self.calc_metrics_core
        count_fn = self.get_batch_weight_core

        fn = lambda b: (core_fn(b), count_fn(b))
        o_specs = vt.get_output_specs_shape_inv(fn, inputs[0][0])
        
        def accum_fn(batches):
            ms, n = zip(*mg.sequential_map(fn, batches, o_specs))
            return vt.weighted_avg(ms, n)
        
        ms, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        ms, _ = vt.weighted_avg(ms, ntoks)

        return ms


        
    def create_train_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
                    
        return (
            dp.ChainableGenerator(
                lambda: zip(dc['source_train'], dc['target_train']))
            .trans(dp.gen_random_sample)
            .trans(dp.gen_line_from_files_multi)
            .trans(dp.gen_skip_empty_line_multi_all)
            .map(lambda l: split_src_ctx_and_main(l, self.pretrain))
            .trans(
                dp.gen_line2IDs_multi,
                (self.vocab_src, self.vocab_src, self.vocab_trg))
            .trans(dp.gen_random_sample, bufsize=bc['shuffle_buffer_size'])
            .trans(
                dp.gen_segment_sort,
                segsize=bc['length_smoothing']['segsize'],
                key=lambda seqs: len(seqs[1]))
            .trans(
                dp.gen_batch_of_capacity_multi,
                bc['size'] // (self.gpus * self.accums),
                capacity_fn=lambda ws, h: sum(max(w**2/200, w) for w in ws)*h)
            .trans(dp.gen_pad_batch_multi)
            .trans(
                dp.gen_random_sample,
                bufsize=bc['length_smoothing']['post_shuf_buf_size'])
        )


    def create_dev_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        
        return (
            dp.ChainableGenerator.zip(
                lambda: dp.gen_line_from_file(dc['source_dev']),
                lambda: dp.gen_line_from_file(dc['target_dev']))
            .map(lambda l: split_src_ctx_and_main(l, self.pretrain))
            .trans(
                dp.gen_line2IDs_multi,
                (self.vocab_src, self.vocab_src, self.vocab_trg))
            .trans(
                dp.gen_batch_of_capacity_multi,
                bc['size'] // (self.gpus * self.accums),
                capacity_fn=lambda ws, h: sum(max(w**2/200, w) for w in ws)*h)
            .trans(dp.gen_pad_batch_multi)
        )
    

    def dataset_from_gen(self, gen):
        w, h = self.accums, self.gpus
        n = w * h

        gen = gen.map(lambda x: dp.list2numpy_nested(x)) \
            .trans(dp.gen_fold, n, (np.zeros([0, 0]),) * 3)
        dataset = super().dataset_from_gen(gen, ((None,)*3,) * n)

        return dataset.map(lambda *x: [x[i:i+w] for i in range(0, n, w)])


    def translate_batch(self, c_x):
        c, x = c_x
        if tf.size(x) > 0:
            # [B, 1, L]
            maxlen = 2 * tf.math.maximum(tf.shape(x)[1], 10)

            paths, scores = self.model.beam_search_decode(
                c, x,
                sos=self.vocab_trg.SOS_ID,
                eos=self.vocab_trg.EOS_ID,
                beam_size=1,
                maxlen=maxlen)
            # [B, L] <- [B, 1, L]
            return paths[:, 0]
        else:
            return x


    def translate_step_(self, inputs):
        """
        Args:
            xs: <[B, L]>[N_gpu * N_accum]
        """
        cxs = [[b[:2] for b in row] for row in inputs]
        ys = [[b[2] for b in row] for row in inputs]

        o_specs = vt.get_output_specs_shape_inv(self.translate_batch, cxs[0][0])

        def accum_fn(cxs):
            return mg.sequential_map(self.translate_batch, cxs, o_specs)
        
        pred = mg.distr_map(accum_fn, cxs)
        return ys, pred


    def translate_step(self, inputs):
        self.translate_step = tf.function(
            self.translate_step_,
            input_signature=[
                (((tf.TensorSpec([None, None], tf.int32),)*3,)*self.accums,)*self.gpus
            ])
        
        return self.translate_step(inputs)


    def check_dataset(self, dataset):
        def batch_gen(b):
            for row in b:
                for c,x,y in row:
                    yield (x, y)

        super().check_dataset(dataset, batch_gen)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='.')
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--accums', type=int)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--mode', type=str,
        choices=['train', 'check_data', 'debug'], default='train')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_eager_function', action='store_true')
    parser.add_argument('--reset_best_score', action='store_true')
    
    args = parser.parse_args(argv)

    basicConfig(level=DEBUG if args.debug else INFO)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    if args.mode == 'train':
        # Configs
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)
        
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)
        
        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)

        # DocTransformer Model
        model = DocTransformer.from_config(model_config, args.pretrain)

        # Vocabulary
        vocab_src, vocab_trg = vt.get_vocabs_from_config(vocab_config)

        # Directory for logging
        logdir = f'{args.dir}'

        trainer = Train(
            args.pretrain,
            model,
            source_vocab=vocab_src,
            target_vocab=vocab_trg,
            train_config=train_config,
            logdir=logdir,
            gpus=args.n_gpus,
            accums=args.accums,
            reset_best_score=args.reset_best_score)
        
        trainer.train()
    elif args.mode == 'check_data':
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab_src, vocab_trg = vt.get_vocabs_from_config(vocab_config)
        
        trainer = Train(
            args.pretrain,
            model=None,
            source_vocab=vocab_src,
            target_vocab=vocab_trg,
            train_config=train_config,
            logdir=None,
            gpus=args.n_gpus,
            accums=args.accums)
        
        train_dataset = trainer.dataset_from_gen(
            trainer.create_train_data_gen()).prefetch(1)
        dev_dataset = trainer.dataset_from_gen(
            trainer.create_dev_data_gen()).prefetch(1)
        print('Train Dataset')
        trainer.check_dataset(train_dataset)
        print('Dev Dataset')
        trainer.check_dataset(dev_dataset)
    elif args.mode == 'debug':
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)

        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab_src, vocab_trg = vt.get_vocabs_from_config(vocab_config)
        
        trainer = Train(
            model=DocTransformer.from_config(model_config),
            source_vocab=vocab_src,
            target_vocab=vocab_trg,
            train_config=train_config,
            logdir=args.dir,
            gpus=args.n_gpus,
            accums=args.accums)
        
        trainer.unit_test(0)
        
    else:
        raise Exception('Invalid parameter')


if __name__ == '__main__':
    main(sys.argv[1:])
