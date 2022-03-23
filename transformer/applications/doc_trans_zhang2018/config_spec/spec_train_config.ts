/**
 * This is the definition of the JSON format configuration of
 * `TrainMultiGPULegacy`.
 * The definition is written in Typescript format.
 */

type Int = number;

type lenSmoothConfig_ = {
        'method': 'segsort',
        'segsize': Int
    };
type LenSmoothConfig = lenSmoothConfig_ & {'post_shuf_buf_size': Int};

export type Config = {
    batch: {
        constraint: 'capacity' | 'size'
        size: Int,
        shuffle_buffer_size: Int | null,
        length_smoothing: LenSmoothConfig
    },
    random_seed: number,
    warm_up_step: Int,
    label_smoothing: number,
    max_step: Int,
    max_epoch: Int,
    summary_interval: Int, // in steps
    early_stopping_criterion: 'bleu' | 'loss',
    early_stopping_patience: Int,
    data: {
        source_train: string[],
        target_train: string[],
        source_dev: string,
        target_dev: string
    }
};
