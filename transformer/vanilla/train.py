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

from .layers import Transformer, label_smoothing
from ..utils import multi_gpu as mg
from ..custom_text_data_pipeline import core as dp
from ..custom_text_data_pipeline.vocabulary import Vocabulary

TSpec = tf.TensorSpec
TCastI = lambda x: tf.cast(x, tf.int32)
TCastI64 = lambda x: tf.cast(x, tf.int64)
TCastF = lambda x: tf.cast(x, tf.float32)


def get_mask(y, dtype=tf.float32):
    return tf.cast(y != 0, dtype)


def sparse_softmax_xent_loss(labels, logits, eps):
    """Sparse softmax crossentropy loss with label smoothing
    An implementation which is theoretically efficient.
    Args:
        labels: [B, L]
        logits: [B, L, V]
        eps: float
    Returns:
        [B, L]
        SXENT(P, Q) = E_P[-log Q]
        P = smooth(onehot(labels))
        log(Q) = log_softmax(logits)
    """
    logp = tf.nn.log_softmax(logits)
    if eps == 0:
        return -tf.gather(logp, labels, batch_dims=2)
    else:
        a = tf.reduce_mean(logp, axis=-1)
        b = tf.gather(logp, labels, batch_dims=2)
        return -(eps * a + (1 - eps) * b)


def loss_additive(logits, y, ls_eps):
    """Unnormalized (not divided by the number of tokens) loss.
    Args:
        logits: <[B, L, V]>
        y: <[B, L]>
    Returns:
        loss: <[], tf.float32>
    """
    #V = tf.shape(logits)[-1]
    #label = label_smoothing(tf.one_hot(y, V), ls_eps)
    #loss = tf.nn.softmax_cross_entropy_with_logits(label, logits)
    loss = sparse_softmax_xent_loss(y, logits, ls_eps)
    return tf.reduce_sum(loss * get_mask(y))


def loss_norm(logits, y, ls_eps):
    """Normalized (per-token) loss.
    Args:
        logits: <[B, L, V]>
        y: <[B, L]>
    Returns:
        loss: <[], tf.float32>
    """
    toks = count_toks(y)
    if toks > 0:
        return loss_additive(logits, y, ls_eps) / toks
    else:
        return tf.constant(0.0)


def count_toks(seqs, dtype=tf.float32):
    return tf.reduce_sum(get_mask(seqs, dtype))


def count_corr(pred, y, dtype=tf.float32):
    pred = tf.cast(pred, tf.int32)
    return tf.reduce_sum(tf.cast((pred == y) &(y != 0), dtype))


def count_corr_from_logits(logits, y):
    """
    Args:
        logits: <[B, L, V]>
    Returns:
        <[B, L]>
    """
    pred = tf.argmax(logits, axis=-1)
    return count_corr(pred, y)


def accuracy(logits, y):
    toks = count_toks(y)
    if toks > 0:
        return count_corr_from_logits(logits, y) / toks
    else:
        return tf.constant(0.0)


def distributed_map_reduce_sum(fn, inputs):
    add_fn = lambda *x: tf.math.add_n(x)
    return map_structure(add_fn , *mg.distr_map(fn, inputs))


def learning_rate(d_model, step, warmup):
    d_model, step, warmup = map(TCastF, (d_model, step, warmup))
    return d_model ** (-0.5) * tf.minimum(step ** -0.5, step * (warmup ** -1.5))


def set_random_seed(seed):
    """Reset the random seed of Python, Tensorflow and Numpy"""
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def weighted_avg(nesteds, weights):
    W = tf.math.add_n(weights)
    if W == 0:
        ws = [0.0 for _ in weights]
    else:
        ws = [w / W for w in weights]
    weighted_nesteds = [
        map_structure(lambda v: v*w, nst) for nst, w in zip(nesteds, ws)]
    fn_add = lambda *x: tf.math.add_n(x)
    return map_structure(fn_add, *weighted_nesteds), W


def get_visible_gpus():
    return len(tf.config.experimental.list_physical_devices('GPU'))
    #len(tf.config.experimental.list_physical_devices('XLA_GPU'))


def get_vocabs_from_config(vocab_config):
    vc = vocab_config
    vocab_src = Vocabulary(
        vc['source_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    vocab_trg = Vocabulary(
        vc['target_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    return vocab_src, vocab_trg


class Stats:
    def __init__(self):
        self.reset()


    def reset(self):
        self.sum = 0
        self.x2 = 0
        self.m = 1e10
        self.M = -1e10
        self.n = 0
        self.mean = 0
        self.var = 0
        self.std = 0


    def update(self, *xs):
        for x in xs:
            self.sum += x
            self.x2 += x ** 2
            self.m = min(self.m, x)
            self.M = max(self.M, x)
            self.n += 1


    def summarize(self):
        self.mean = self.sum / self.n
        self.var = self.x2 / self.n - self.mean ** 2
        self.std = self.var ** 0.5
        return {
            'mean': self.mean,
            'sum': self.sum,
            'min': self.m,
            'max': self.M,
            'var': self.var,
            'std': self.std,
            'n': self.n
        }


class StatsCorrXY:
    def __init__(self):
        self.reset()


    def reset(self):
        self.X = np.zeros(2)
        self.X2 = np.zeros(2)
        self.xy = 0
        self.n = 0


    def update(self, xs, ys):
        X = np.stack([xs, ys], axis=1)
        self.X += X.sum(axis=0)
        self.X2 += (X ** 2).sum(axis=0)
        self.xy += X.prod(axis=1).sum()
        self.n += len(X)


    def summarize(self):
        M = self.X / self.n
        V = self.X2 / self.n - M ** 2
        nume = self.xy / self.n - M.prod()
        deno = V.prod() ** 0.5
        return {
            'corr': nume / deno,
            'means': M,
            'stds': V ** 0.5
        }


def get_shape_inv_spec(t):
    rank = len(t.shape.as_list())
    return tf.TensorSpec([None]*rank, t.dtype)


def deco_function_oneshot_shape_inv(f=None, argc=1):
    if f is None:
        return lambda f_: deco_function_oneshot_shape_inv(f_, argc)

    graph_fn = None

    def f_(*args):
        nonlocal graph_fn
        if graph_fn is None:
            if len(args) == argc + 1:
                inputs = args[1:]
                wrapped = lambda *args_: f(args[0], *args_)
            else:
                assert len(args) == argc
                inputs = args
                wrapped = lambda *args_: f(*args_)

            specs = nest.map_structure(get_shape_inv_spec, inputs)
            graph_fn = tf.function(wrapped, input_signature=specs)

        return graph_fn(*args[-argc:])

    return f_


def get_output_dtypes(fn, *args, **kwargs):
    outs = fn(*args, **kwargs)
    return nest.map_structure(lambda x: x.dtype, outs)


def get_output_specs_shape_inv(fn, *args, **kwargs):
    outs = fn(*args, **kwargs)
    return nest.map_structure(lambda x: get_shape_inv_spec(x), outs)


class Train:
    def __init__(
            self,
            model,
            source_vocab,
            target_vocab,
            train_config,
            logdir,
            reset_best_score=False,
            ckpt_on_summary=False):
        self.logdir = logdir

        self.model = model

        self.reset_best_score = reset_best_score
        
        self.ckpt_on_summary = ckpt_on_summary

        self.train_config = train_config

        self.vocab_src = source_vocab
        self.vocab_trg = target_vocab

        def calc_loss_(logits, y, loss):
            return loss

        def calc_accuracy_(logits, y, loss):
            return accuracy(logits, y)

        self.metrics = {
            'loss': {
                'calc': calc_loss_,
                'train_mean': keras.metrics.Mean(),
                'dev_mean': keras.metrics.Mean()
            },
            'accuracy': {
                'calc': calc_accuracy_,
                'train_mean': keras.metrics.Mean(),
                'dev_mean': keras.metrics.Mean()
            }
        }

        # Dataset generator pipeline settings
        pfn = self.pipeline_fns = {}
        bc = self.train_config['batch']

        # How to read lines from multiple files (default: no interleaving)
        pfn['line_from_files_multi'] = dp.gen_line_from_files_multi

        # Length smoothing
        if bc['length_smoothing'] is None:
            pfn['length_smoothing'] = lambda x: x
            pfn['post_ls_shuffle'] = lambda x: x
        elif bc['length_smoothing']['method'] == 'segsort':
            pfn['length_smoothing'] = lambda x: dp.gen_segment_sort(
                x,
                segsize=bc['length_smoothing']['segsize'],
                key=lambda seqs: len(seqs[0]))
            pfn['post_ls_shuffle'] = lambda x: dp.gen_random_sample(
                x,
                bufsize=bc['length_smoothing']['post_shuf_buf_size'])
        else:
            assert False
        
        # Batching
        if bc['constraint'] == 'size':
            pfn['batching'] = lambda x: dp.gen_batch_multi(x, bc['size'])
        else:
            pfn['batching'] = \
                lambda x: dp.gen_batch_of_capacity_multi(x, bc['size'])


    def calc_metrics(self, batch):
        x, y = batch
        if tf.size(x) > 0:
            y_i, y_o = y[:, :-1], y[:, 1:]
            logits = self.model(x, y_i, training=False)
            loss = loss_norm(
                logits, y_o, ls_eps=self.train_config['label_smoothing'])
            metrics = {k: v['calc'](logits, y_o, loss)
                for k, v in self.metrics.items()}
        else:
            metrics = {k: 0.0 for k in self.metrics.keys()}

        return metrics


    def calc_grad_metrics(self, batch):
        """Compute gradient, metrics and #tokens given a batch.
        Args:
            batch: (x: <[B, L_x]>, y: <[B, L_y]>)
        Returns:
            (grad: Gradient, metrics: list<Tensor>, n_tokens: tf.int32)
        """
        x, y = batch
        tc = self.train_config
        y_i, y_o = y[:, :-1], y[:, 1:]
        with tf.GradientTape() as tape:
            logits = self.model(x, y_i, training=True)
            loss = loss_norm(logits, y_o, ls_eps=tc['label_smoothing'])

        grad = tape.gradient(
            loss,
            self.model.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        metrics = {k: v['calc'](logits, y_o, loss)
            for k, v in self.metrics.items()}

        return grad, metrics
    

    def get_batch_weight(self, batch):
        x, y = batch
        return count_toks(y[:, 1:])


    @deco_function_oneshot_shape_inv
    def train_step(self, inputs):
        g, metrics = self.calc_grad_metrics(inputs)

        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))
        
        ntoks = self.get_batch_weight(inputs)
        for k,v in self.metrics.items():
            v['train_mean'].update_state(metrics[k], ntoks)
    

    @deco_function_oneshot_shape_inv
    def dev_step(self, inputs):
        metrics = self.calc_metrics(inputs)

        ntoks = self.get_batch_weight(inputs)
        for k,v in self.metrics.items():
            v['dev_mean'].update_state(metrics[k], ntoks)
    

    def update_dev_metrics(self, dev_dataset):
        for metric in self.metrics.values():
            metric['dev_mean'].reset_states()

        for data in dev_dataset:
            self.dev_step(data)
    

    def write_dev_metrics(self, writer, step):
        with writer.as_default():
            for name, m in self.metrics.items():
                tf.summary.scalar(f'{name}_dev', m['dev_mean'].result(), step=TCastI64(step))


    def write_and_reset_train_metrics(self, writer, step):
        with writer.as_default():
            for name, m in self.metrics.items():
                tf.summary.scalar(f'{name}_train', m['train_mean'].result(), step=TCastI64(step))
                m['train_mean'].reset_states()


    def translate_batch(self, x):
        """
        Args:
            x: [B, L]
        Returns:
            [B, L_OUT]
        """
        if tf.size(x) > 0:
            # [B, 1, L]
            maxlen = 2 * tf.math.maximum(tf.shape(x)[1], 10)

            paths, scores = self.model.beam_search_decode_with_prefix(
                x,
                prefix_or_sos=self.vocab_trg.SOS_ID,
                eos=self.vocab_trg.EOS_ID,
                beam_size=1,
                maxlen=maxlen)
            # [B, L] <- [B, 1, L]
            return paths[:, 0]
        else:
            return x


    @deco_function_oneshot_shape_inv
    def translate_step(self, inputs):
        """
        Args:
            inputs: ([B, L_x], [B, L_y])
        """
        x, y = inputs
        return y, self.translate_batch(x)


    def compute_bleu(self, dataset):
        """Compute BLEU on subwords"""

        refs, hyps = [], []

        for batch in dataset:
            y, pred = self.translate_step(batch)
            for y_ in nest.flatten(y):
                refs.extend(self.vocab_trg.IDs2text(y_.numpy()))
            for pred_ in nest.flatten(pred):
                hyps.extend(self.vocab_trg.IDs2text(pred_.numpy()))
        
        samples = '\n'.join(
            map(lambda r_o: f'[Ref] {r_o[0]}\n[Out] {r_o[1]}',
                itertools.islice(zip(refs, hyps), 5)))
        logger.debug('First 5 lines of reference and translation\n' + samples)

        refs = [[line.split()] for line in refs]
        hyps = [line.split() for line in hyps]

        return corpus_bleu(refs, hyps)


    def create_train_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        pfn = self.pipeline_fns
                    
        return (
            dp.ChainableGenerator(
                lambda: zip(dc['source_train'], dc['target_train']))
            .trans(dp.gen_random_sample)
            .trans(pfn['line_from_files_multi'])
            .trans(dp.gen_line2IDs_multi, (self.vocab_src, self.vocab_trg))
            .trans(dp.gen_random_sample, bufsize=bc['shuffle_buffer_size'])
            .trans(pfn['length_smoothing'])
            .trans(pfn['batching'])
            .trans(dp.gen_pad_batch_multi)
            .trans(pfn['post_ls_shuffle'])
        )


    def create_dev_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        pfn = self.pipeline_fns
        
        return (
            dp.ChainableGenerator.zip(
                lambda: dp.gen_line_from_file(dc['source_dev']),
                lambda: dp.gen_line_from_file(dc['target_dev']))
            .trans(dp.gen_line2IDs_multi, (self.vocab_src, self.vocab_trg))
            .trans(pfn['batching'])
            .trans(dp.gen_pad_batch_multi)
        )
    

    def dataset_from_gen(self, gen, structure=None):
        structure = (None, None) if structure is None else structure
        dtype = map_structure(lambda x: tf.int32, structure)
        shape = map_structure(lambda x: tf.TensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def unit_test(self, i):
        ckpt = tf.train.Checkpoint(model=self.model)
        lst_ckpt = tf.train.latest_checkpoint(f'{self.logdir}/checkpoint_best')
        logger.debug('Restoring' f'{lst_ckpt}')
        ckpt.restore(lst_ckpt)
        logger.debug('Restored')
        if i == 0:
            dataset = self.dataset_from_gen(self.create_dev_data_gen())

            refs, hyps = [], []

            t = time.time()
            for batch in dataset:
                y, pred = self.translate_step(batch)
                hyps.extend(self.vocab_trg.IDs2text(pred.numpy()))
                refs.extend(self.vocab_trg.IDs2text(y.numpy()))
                for hyp in hyps:
                    print(hyp)
                t, dt = time.time(), time.time() - t
                logger.debug(dt)

            
            refs = [[line.split()] for line in refs]
            hyps = [line.split() for line in hyps]
            logger.info(corpus_bleu(refs, hyps))


    def train(self):
        tc = self.train_config

        # Random
        set_random_seed(tc['random_seed'])
        rnd = random.Random(tc['random_seed'])

        # Dataset
        train_dataset = \
            self.dataset_from_gen(self.create_train_data_gen()).prefetch(1)
        dev_dataset = \
            self.dataset_from_gen(self.create_dev_data_gen()).prefetch(1)

        # Step Counters
        epoch = tf.Variable(0, dtype=tf.int32)
        step = tf.Variable(0, dtype=tf.int32)
        loc_step = tf.Variable(0, dtype=tf.int32) # Steps reset in every epoch

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            lambda: learning_rate(self.model.d_model, step, tc['warm_up_step']),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        # Early Stopping
        best_epoch = tf.Variable(0, dtype=tf.int32)
        best_score = tf.Variable(-1e10)

        # Checkpoint and Managers

        # Main checkpoint
        ckpt = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            loc_step=loc_step,
            optimizer=self.optimizer,
            model=self.model,
            best_epoch=best_epoch,
            best_score=best_score
        )

        # Main manager
        manager = tf.train.CheckpointManager(
            ckpt,
            directory=f'{self.logdir}/checkpoint',
            max_to_keep=tc.get('ckpt_max_to_keep', 1),
            step_counter=step)
        
        # Manger for long-term history
        manager_hist = tf.train.CheckpointManager(
            ckpt,
            directory=f'{self.logdir}/checkpoint_history',
            max_to_keep=None,
            step_counter=step)

        # Checkpoint for recording the best epoch
        ckpt_best = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            model=self.model
        )

        manager_best = tf.train.CheckpointManager(
            ckpt_best,
            directory=f'{self.logdir}/checkpoint_best',
            max_to_keep=3,
            step_counter=step)
        
        if manager.latest_checkpoint:
            logger.info(f'Restoring from {manager.latest_checkpoint}')
            ckpt.restore(manager.latest_checkpoint)
            logger.info(
                f'Restored\n'
                f'Epoch: {epoch.numpy()},\n'
                f'Step: {step.numpy()}\n'
                f'Best Checkpoint: \n'
                f'\tEpoch: {best_epoch.numpy()}\n'
                f'\tScore: {best_score.numpy()}\n'
            )
        else:
            logger.info('Checkpoint was not found')

        if self.reset_best_score:
            logger.warn('\n\nReset best score\n')
            best_score.assign(-1e10)

        start_epoch = epoch.numpy()
        start_loc_step = loc_step.numpy()
        
        # Summary
        writer = tf.summary.create_file_writer(f'{self.logdir}/summary')

        # Training Loop
        logger.debug('Train Loop Starts')
        for epoch_ in range(tc['max_epoch']):
            if epoch_ < start_epoch:
                continue
            
            set_random_seed(rnd.randrange(0xFFFF))
            
            # Epoch Loop
            t = time.time()
            for loc_step_, data in enumerate(train_dataset):
                if loc_step_ < start_loc_step:
                    continue
                elif loc_step_ == start_loc_step:
                    start_loc_step = -1

                
                self.train_step(data)

                step.assign_add(1)

                t_ = time.time()
                t, dt = t_, t_ - t
                sys.stdout.write(f'Step: {step.numpy()}, Time elapsed: {dt}\n')
                sys.stdout.flush()

                # Summary
                if step.numpy() % tc['summary_interval'] == 0:
                    self.update_dev_metrics(dev_dataset)
                    self.write_dev_metrics(writer, step)
                    self.write_and_reset_train_metrics(writer, step)
                    
                    if self.ckpt_on_summary:
                        logger.info('Saving...')
                        manager.save(step)

                
            epoch.assign_add(1)
            loc_step.assign(0)

            # Epoch Summary
            # Basic summary
            self.update_dev_metrics(dev_dataset)
            self.write_dev_metrics(writer, step)
            loss = self.metrics['loss']['dev_mean'].result().numpy()

            # BLEU
            logger.info('Computing BLEU')
            bleu = self.compute_bleu(dev_dataset)
            with writer.as_default():
                tf.summary.scalar('BLEU', bleu, step=TCastI64(step))

            logger.info(f'Epoch {epoch.numpy()}, Loss: {loss}, BLEU: {bleu}')
            
            # Early Stopping
            if tc['early_stopping_criterion'] == 'loss':
                score_ = -loss
            else:
                score_ = bleu

            logger.debug(
                f'Last Best: {best_score.numpy()}, This time: {score_}')

            should_early_stop = False
            if score_ > best_score.numpy():
                best_score.assign(score_)
                best_epoch.assign(epoch)

                logger.info('Updating the best checkpoint')
                manager_best.save(step)
            elif epoch - best_epoch > tc['early_stopping_patience']:
                should_early_stop = True

            # Checkpoint
            logger.info('Checkpointing')
            manager.save(step)

            # History
            _t = epoch.numpy()
            if int(_t ** 0.5) ** 2 == _t:
                logger.info('Saving as long-term checkpoint')
                manager_hist.save(step)

            if should_early_stop:
                logger.info('Early Stopping')
                break
    

    def check_dataset(self, dataset, xy_generator=None):
        @tf.function(input_signature=[dataset.element_spec])
        def toks_(batch):
            return tf.math.add_n([count_toks(x) for x in nest.flatten(batch)])

        @tf.function(input_signature=[dataset.element_spec])
        def sents_(batch):
            return tf.math.add_n([tf.shape(x)[0] for x in nest.flatten(batch)])

        @tf.function(input_signature=[dataset.element_spec])
        def capacity_(batch):
            return tf.math.add_n([tf.size(x) for x in nest.flatten(batch)])

        @tf.function(input_signature=[dataset.element_spec])
        def longest_(batch):
            lens = [tf.shape(x)[1] for x in nest.flatten(batch)]
            return tf.math.reduce_max(lens)
        
        @tf.function(input_signature=[dataset.element_spec])
        def lens(batch):
            len_fn = lambda x: tf.reduce_sum(get_mask(x), axis=1)
            lens = nest.map_structure(len_fn, batch)
            if xy_generator is not None:
                xs, ys = zip(*xy_generator(lens))
            else:
                lens = nest.flatten(lens)
                xs, ys = zip(*[lens[i: i + 2] for i in range(0, len(lens), 2)])
            x = tf.concat(xs, axis=0)
            y = tf.concat(ys, axis=0)

            # [B], [B]
            return x, y
            

        metrics = ['Sec', 'Tokens', 'Sents', 'Capacity','Longest']
        stats = [Stats() for i in range(len(metrics))]
        len_corr = StatsCorrXY()
        
        last_t = time.time()
        i = 0
        for data in dataset:
            t = time.time()
            last_t, dt = t, t - last_t
            scores = [
                dt, toks_(data).numpy(), sents_(data).numpy(),
                capacity_(data).numpy(), longest_(data).numpy()]
            for sts, score in zip(stats, scores):
                sts.update(score)

            l_x, l_y = lens(data)
            len_corr.update(l_x.numpy(), l_y.numpy())

            i += 1
            if i % 100 == 0:
                print(i)

        print(f'Steps: {i}')
        for m, sts in zip(metrics, stats):
            res = sts.summarize()
            print(f'{m}/Step')
            for label, score in res.items():
                print(f'{label}: {score}')
            print()

        print('Corr')
        print(len_corr.summarize())
        print()


class TrainMultiGPULegacy(Train):
    def __init__(self, *args, gpus=None, accums=None,
            split_type='small_minibatch', **kwargs):
        super().__init__(*args, **kwargs)

        vis_gpus = get_visible_gpus()
        self.gpus = vis_gpus if gpus is None else gpus
        logger.debug(f'Number of GPUs: {self.gpus}')

        self.accums = 1 if accums is None else accums

        self.split_type = split_type

        if split_type == 'small_minibatch':
            pfn = self.pipeline_fns
            n = self.gpus * self.accums
            bc = self.train_config['batch']
            if bc['constraint'] == 'size':
                pfn['batching'] = lambda x: dp.gen_batch_multi(x, bc['size'] // n)
            else:
                pfn['batching'] = \
                    lambda x: dp.gen_batch_of_capacity_multi(x, bc['size'] // n)
        elif split_type == 'divide_batch':
            pass
        else:
            raise Exception('Invalid parameter')

    

    def dataset_from_gen(self, gen):
        w, h = self.accums, self.gpus
        n = w * h

        if self.split_type == 'small_minibatch':
            gen = gen.map(lambda x: dp.list2numpy_nested(x)) \
                .trans(dp.gen_fold, n, (np.zeros([0, 0]),) * 2)
            dataset = super().dataset_from_gen(gen, ((None,)*2,) * n)
        else:
            dataset = super().dataset_from_gen(gen) \
                .map(lambda *x: mg.non_even_split(x, n))

        return dataset.map(lambda *x: [x[i:i+w] for i in range(0, n, w)])


    def calc_grad_metrics(self, inputs):
        """
        Args:
            inputs:
                list<pair<x, y>, N_accum * N_gpu>
                x: Tensor<[B, L_src], int32>
                y: Tensor<[B, L_trg], int32>
        """
        core_fn = super().calc_grad_metrics
        count_fn = lambda b: count_toks(b[1][:, 1:], tf.float32)

        fn = lambda b: (core_fn(b), count_fn(b))
        o_specs = get_output_specs_shape_inv(fn, inputs[0][0])

        def accum_fn(batches):
            g_ms, ntoks = zip(*mg.sequential_map(fn, batches, o_specs))
            return weighted_avg(g_ms, ntoks)

        g_ms, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        g_ms, _ = weighted_avg(g_ms, ntoks)

        return g_ms


    def calc_metrics(self, inputs):
        core_fn = super().calc_metrics
        count_fn = lambda b: count_toks(b[1][:, 1:], tf.float32)

        fn = lambda b: (core_fn(b), count_fn(b))
        o_specs = get_output_specs_shape_inv(fn, inputs[0][0])
        
        def accum_fn(batches):
            ms, n = zip(*mg.sequential_map(fn, batches, o_specs))
            return weighted_avg(ms, n)
        
        ms, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        ms, _ = weighted_avg(ms, ntoks)

        return ms


    def get_batch_weight(self, batch):
        ys = nest.flatten([[b[1] for b in row] for row in batch])
        return tf.math.add_n([count_toks(y[:, 1:]) for y in ys])
    

    def translate_step_(self, inputs):
        """
        Args:
            xs: <[B, L]>[N_gpu * N_accum]
        """
        xs = [[b[0] for b in row] for row in inputs]
        ys = [[b[1] for b in row] for row in inputs]
        spec = TSpec([None, None], tf.int32),

        o_specs = get_output_specs_shape_inv(self.translate_batch, xs[0][0])

        def accum_fn(xs):
            return mg.sequential_map(self.translate_batch, xs, o_specs)
        
        pred = mg.distr_map(accum_fn, xs)
        return ys, pred


    def translate_step(self, inputs):
        self.translate_step = tf.function(
            self.translate_step_,
            input_signature=[
                (((TSpec([None, None], tf.int32),)*2,)*self.accums,)*self.gpus
            ])
        
        return self.translate_step(inputs)



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='.')
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--accums', type=int)
    parser.add_argument('--mode', type=str,
        choices=['train', 'check_data', 'debug'], default='train')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_base_class', action='store_true')
    parser.add_argument('--debug_post_split', action='store_true')
    parser.add_argument('--debug_eager_function', action='store_true')
    parser.add_argument('--reset_best_score', action='store_true')
    parser.add_argument('--ckpt_on_summary', action='store_true')
    
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

        # Transformer Model
        model = Transformer.from_config(model_config)

        # Vocabulary
        vocab_src, vocab_trg = get_vocabs_from_config(vocab_config)

        # Directory for logging
        logdir = f'{args.dir}'

        if args.debug_base_class:
            trainer = Train(
                model,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=logdir,
                reset_best_score=args.reset_best_score,
                ckpt_on_summary=args.ckpt_on_summary)
        else:
            trainer = TrainMultiGPULegacy(
                model,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=logdir,
                gpus=args.n_gpus,
                accums=args.accums,
                split_type='divide_batch' \
                    if args.debug_post_split else 'small_minibatch',
                reset_best_score=args.reset_best_score,
                ckpt_on_summary=args.ckpt_on_summary)
        
        trainer.train()
    elif args.mode == 'check_data':
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab_src, vocab_trg = get_vocabs_from_config(vocab_config)
        
        if args.debug_base_class:
            trainer = Train(
                model=None,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=None)
        else:
            trainer = TrainMultiGPULegacy(
                model=None,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=None,
                gpus=args.n_gpus,
                accums=args.accums,
                split_type='divide_batch' \
                    if args.debug_post_split else 'small_minibatch')
        
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
        
        vocab_src, vocab_trg = get_vocabs_from_config(vocab_config)
        
        if args.debug_base_class:
            trainer = Train(
                model=Transformer.from_config(model_config),
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=args.dir)
        else:
            trainer = TrainMultiGPULegacy(
                model=Transformer.from_config(model_config),
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=args.dir,
                gpus=args.n_gpus,
                accums=args.accums,
                split_type='divide_batch' \
                    if args.debug_post_split else 'small_minibatch')
        
        trainer.unit_test(0)
        
    else:
        raise Exception('Invalid parameter')


if __name__ == '__main__':
    main(sys.argv[1:])
