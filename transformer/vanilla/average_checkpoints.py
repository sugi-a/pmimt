import sys
import json
import argparse
from logging import getLogger, basicConfig, DEBUG
logger = getLogger()

import tensorflow as tf

from .layers import Transformer

def average_checkpoints(
        ckpt_paths,
        create_model_fn,
        dst_ckpt_prefix,
        model_name="model"
    ):
    n = len(ckpt_paths)

    # Source models
    models = [create_model_fn() for i in range(n)]

    # Destination model
    dst_model = create_model_fn()

    # Instanciate variables
    logger.info('Instanciate variables')

    x = tf.zeros([2, 2], dtype=tf.int32)
    y = tf.zeros([2, 2], dtype=tf.int32)
    for model in models:
        model(x, y, training=False)
    dst_model(x, y, training=False)

    logger.info('Instanciated variables')

    logger.debug('\n\nIn Model' + '\n'.join(map(lambda x: x.name, models[0].variables)))
    logger.debug('\n\nIn Checkpoint\n' + '\n'.join(map(str, tf.train.list_variables(ckpt_paths[0]))))

    # Checkpoints
    #ckpts = [tf.train.Checkpoint(**{model_name: model}) for model in models]
    ckpts = [tf.train.Checkpoint(model=model) for model in models]
    #dst_ckpt = tf.train.Checkpoint(**{model_name: dst_model})
    dst_ckpt = tf.train.Checkpoint(model=dst_model)

    # Load checkponts
    for ckpt, path in zip(ckpts, ckpt_paths):
        logger.info(f'Loading {path}')
        ckpt.restore(path).assert_existing_objects_matched()
        logger.info(f'Loaded {path}')

    logger.info('Averaging')
    for dst_var, src_vars in zip(
            dst_model.weights,
            zip(*(m.weights for m in models))
        ):
        dst_var.assign(tf.add_n(src_vars) / n)

    logger.info(f'Saving {dst_ckpt_prefix}')
    dst_ckpt.save(dst_ckpt_prefix)
    logger.info(f'Saved {dst_ckpt_prefix}')


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--dir', '-d', type=str, default='.')
    p.add_argument('--model_name', type=str, default='model')
    p.add_argument('--dest', required=True, type=str)
    p.add_argument('--debug', action='store_true')
    p.add_argument('paths', type=str, nargs='+')

    args = p.parse_args(argv)
    
    if args.debug:
        basicConfig(level=DEBUG)
    
    # Config
    with open(f'{args.dir}/model_config.json') as f:
        model_config = json.load(f)

    logger.info('Start')
    average_checkpoints(
        args.paths,
        lambda: Transformer.from_config(model_config),
        args.dest,
        model_name=args.model_name)


if __name__ == '__main__':
    main()
