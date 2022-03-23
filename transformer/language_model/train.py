from logging import getLogger; logger = getLogger(__name__)
from logging import DEBUG, INFO, basicConfig
import os, sys, argparse, time, json
import itertools
import random
from collections import deque

import tensorflow as tf
from tensorflow import keras, nest
import numpy as np

from .layers import DecoderLanguageModel 
from ..utils import multi_gpu as mg
from ..custom_text_data_pipeline import core as dp
from ..custom_text_data_pipeline.vocabulary import Vocabulary
from ..vanilla.train import \
    sparse_softmax_xent_loss, \
    get_mask, \
    deco_function_oneshot_shape_inv, \
    get_visible_gpus, \
    Stats, \
    count_toks, \
    get_output_specs_shape_inv, \
    weighted_avg, \
    learning_rate
from .datasetloader import \
    create_simple_batch_generator, \
    create_front_aligned_doc_segment_generator


TShape = tf.TensorShape


def get_vocabs_from_config(config):
    return Vocabulary(
        vocab_file=config['dict'],
        PAD_ID=config['PAD_ID'],
        EOS_ID=config['EOS_ID'],
        UNK_ID=config['UNK_ID'],
        SOS_ID=config['SOS_ID'])


class Train:
    def __init__(
            self,
            model,
            vocab,
            train_config,
            logdir,
            gpus=None,
            accums=None):
        self.logdir = logdir
        self.model = model
        self.train_config = train_config
        self.vocab = vocab

        self.dev_loss = keras.metrics.Mean()

        vis_gpus = get_visible_gpus()
        self.gpus = vis_gpus if gpus is None else gpus
        logger.info(f'Number of GPUs: {self.gpus}')

        self.accums = 1 if accums is None else accums


    def calc_loss(self, x, training):
        x_i, x_o = x[:, :-1], x[:, 1:]

        mask = get_mask(x_o)
        ntoks = tf.reduce_sum(mask)
            
        if ntoks > 0:
            lgts = self.model(x_i, training=training)
        
            losses = sparse_softmax_xent_loss(
                x_o, lgts, self.train_config['label_smoothing'])

            loss = tf.reduce_sum(losses * mask) / ntoks
        else:
            loss = tf.constant(0.0)
        
        return loss


    def calc_grad(self, x):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(x, True)
        
        grad = tape.gradient(loss, self.model.trainable_variables)
        
        return grad 
    

    def get_batch_weight(self, batch):
        return count_toks(batch[:, 1:])

    
    @deco_function_oneshot_shape_inv
    def train_step(self, inputs):
        count_fn = lambda b: count_toks(b[:, 1:], tf.float32)
        map_fn = lambda b: (self.calc_grad(b), count_fn(b))

        def reduce_fn(*g_ntoks):
            gs, ntoks = zip(*g_ntoks)
            g, ntok = weighted_avg(gs, ntoks)
            return (g, ntok)

        def accum_fn(batches):
            g, ntok = mg.sequential_map_reduce(map_fn, reduce_fn, batches)
            return g, ntok

        gs, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        g, _ = weighted_avg(gs, ntoks)
        
        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))
        

    @deco_function_oneshot_shape_inv
    def dev_step(self, inputs):
        count_fn = lambda b: count_toks(b[:, 1:], tf.float32)
        map_fn = lambda b: (self.calc_loss(b, False), count_fn(b))

        def reduce_fn(*o_ntoks):
            return weighted_avg(*zip(*o_ntoks))

        def accum_fn(batches):
            return mg.sequential_map_reduce(map_fn, reduce_fn, batches)

        os, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        o, ntok = weighted_avg(os, ntoks)

        self.dev_loss.update_state(o, ntok)


    def update_dev_metrics(self, dev_dataset):
        self.dev_loss.reset_states()

        for data in dev_dataset:
            self.dev_step(data)


    def write_dev_metrics(self, writer, step):
        with writer.as_default():
            tf.summary.scalar(
                'dev_loss',
                self.dev_loss.result(),
                step=tf.cast(step, tf.int64))
    

    def create_train_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        sc = bc['sampling']

        w, h = self.accums, self.gpus
        n = w * h
        capacity = bc['capacity'] // n

        if sc['mode'] == 'normal':
            return create_simple_batch_generator(
                files=dc['train'],
                vocab=self.vocab,
                stochastic=True,
                batch_capacity=capacity,
                shuf_buf_size=bc['shuf_buf_size'],
                length_smoothing=sc['length_smoothing'],
                batch_shuf_buf_size=sc['batch_shuf_buf_size']) \
            .map(dp.list2numpy_nested) \
            .trans(dp.gen_fold, n, np.zeros([0, 0])) \
            .map(lambda b: tuple(b[i: i+w] for i in range(0, n, w)))
        elif sc['mode'] == 'front_aligned_segs_from_docs':
            return create_front_aligned_doc_segment_generator(
                files=dc['train'],
                vocab=self.vocab,
                stochastic=True,
                max_window_size=sc['max_window_size'],
                min_window_size=sc['min_window_size'],
                min_stride=sc['min_stride'],
                rand_extra_stride=sc['rand_extra_stride'],
                capacity=capacity,
                shuf_buf_size=bc['shuf_buf_size']) \
            .map(dp.list2numpy_nested) \
            .trans(dp.gen_fold, n, np.zeros([0, 0])) \
            .map(lambda b: tuple(b[i: i+w] for i in range(0, n, w)))


    def create_dev_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        sc = bc['sampling']

        w, h = self.accums, self.gpus
        n = w * h

        capacity = bc['capacity'] // n

        if sc['mode'] == 'normal':
            return create_simple_batch_generator(
                files=[dc['dev']],
                vocab=self.vocab,
                stochastic=False,
                batch_capacity=capacity,
                length_smoothing=sc['length_smoothing']) \
            .map(dp.list2numpy_nested) \
            .trans(dp.gen_fold, n, np.zeros([0, 0])) \
            .map(lambda b: tuple(b[i: i+w] for i in range(0, n, w)))
        elif sc['mode'] == 'front_aligned_segs_from_docs':
            return create_front_aligned_doc_segment_generator(
                files=[dc['dev']],
                vocab=self.vocab,
                stochastic=False,
                max_window_size=sc['max_window_size'],
                min_window_size=sc['min_window_size'],
                min_stride=sc['min_stride'],
                rand_extra_stride=0,
                capacity=capacity) \
            .map(dp.list2numpy_nested) \
            .trans(dp.gen_fold, n, np.zeros([0, 0])) \
            .map(lambda b: tuple(b[i: i+w] for i in range(0, n, w)))
        else:
            raise


    def dataset_from_gen(self, gen):
        w, h = self.accums, self.gpus
        n = w * h
        structure = ((None,) * w,) * h
        dtype = nest.map_structure(lambda x: tf.int32, structure)
        shape = nest.map_structure(lambda x: TShape([None,]*2), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def train(self):
        tc = self.train_config

        # Random
        def set_random_seed(seed):
            """Reset the random seed of Python, Tensorflow and Numpy"""
            random.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)

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
            lambda: learning_rate(self.model.d_model, step, tc['warm_up_steps']),
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
            max_to_keep=1,
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

                # Print processing speed
                t_ = time.time()
                t, dt = t_, t_ - t
                sys.stdout.write(f'Step: {step.numpy()}, Time elapsed: {dt}\n')
                sys.stdout.flush()

                # Summary
                if step.numpy() % tc['summary_interval'] == 0:
                    self.update_dev_metrics(dev_dataset)
                    self.write_dev_metrics(writer, step)
                
            epoch.assign_add(1)
            loc_step.assign(0)

            # Epoch Summary
            # Basic summary
            self.update_dev_metrics(dev_dataset)
            self.write_dev_metrics(writer, step)
            loss = self.dev_loss.result()
            logger.info(f'Epoch {epoch.numpy()}, Loss: {loss}')
            
            # Early Stopping
            score = -loss
            logger.debug(
                f'Last Best: {best_score.numpy()}, This time: {score}')

            should_early_stop = False
            if score > best_score.numpy():
                best_score.assign(score)
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


    def check_dataset(self, dataset):
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

        metrics = ['Sec', 'Tokens', 'Sents', 'Capacity','Longest']
        stats = [Stats() for i in range(len(metrics))]
        
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
            i += 1
            if i % 100 == 0:
                print(i)
                sys.stdout.flush()

        print(f'Steps: {i}')
        for m, sts in zip(metrics, stats):
            res = sts.summarize()
            print(f'{m}/Step')
            for label, score in res.items():
                print(f'{label}: {score}')
            print()


    def debug(self):
        gen = self.create_train_data_gen()
        for d in gen():
            a = d


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='.')
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--accums', type=int)
    parser.add_argument('--mode', type=str,
        choices=['train', 'check_data', 'debug'], default='train')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_eager_function', action='store_true')
    
    args = parser.parse_args(argv)

    basicConfig(level=DEBUG if args.debug else INFO)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    if args.mode == 'train' or args.mode == 'debug':
        # Configs
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)
        
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)
        
        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)

        # Transformer Model
        model = DecoderLanguageModel.from_config(model_config)

        # Vocabulary
        vocab = get_vocabs_from_config(vocab_config)

        # Directory for logging
        logdir = f'{args.dir}'

        trainer = Train(
            model,
            vocab=vocab,
            train_config=train_config,
            logdir=logdir,
            gpus=args.n_gpus,
            accums=args.accums)
        
        if args.mode == 'train':
            trainer.train()
        else:
            trainer.debug()

    elif args.mode == 'check_data':
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab = get_vocabs_from_config(vocab_config)
        
        trainer = Train(
            model=None,
            vocab=vocab,
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


if __name__ == '__main__':
    main(sys.argv[1:])
