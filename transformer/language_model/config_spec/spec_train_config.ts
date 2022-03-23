type Int = number;
type Float = number;

type BatchSampling = {
        mode: 'normal',
        length_smoothing: Int | null,
        batch_shuf_buf_size: Int | null
    } | {
        mode: 'front_aligned_segs_from_docs',
        max_window_size: Int,
        min_window_size: Int,
        min_stride: Int,
        rand_extra_stride: Int
    }

export type Config = {
    batch: {
        sampling: BatchSampling,
        shuf_buf_size: Int,
        capacity: Int
    },
    random_seed: Int,
    warm_up_steps: Int,
    label_smoothing: Float,
    max_step: Int,
    max_epoch: Int,
    summary_interval: Int, // in steps
    early_stopping_patience: Int, // in epochs
    data: {
        train: string[],
        dev: string
    }
};
