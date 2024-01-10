from datasets import IterableDataset, interleave_datasets
import librosa
import numpy as np
from functools import partial
import wave
import soundfile


### copied from feeeeedback
def reshape_data(tensor, num_channels, slice_length, step_back):
    # Create the start index for each channel
    seq_length = tensor.shape[-1]
    start_indices = np.arange(num_channels) * step_back
    start_indices = seq_length - slice_length - start_indices

    # Create a 2D index matrix
    index_matrix = np.expand_dims(start_indices, axis=1) + np.arange(slice_length)

    # Index into the original tensor to create the new shape
    reshaped_tensor = tensor[..., index_matrix]
    return reshaped_tensor


def make_input_chunk(input_d0, sample_width, dilation, zone_size, window, shift):
    input_d0 = np.array(input_d0)
    input_d1 = input_d0[..., (sample_width - 1) % (dilation * 2) :: dilation * 2]
    input_d2 = input_d0[..., (sample_width - 1) % (dilation * 3) :: dilation * 3]
    input_d3 = input_d0[..., (sample_width - 1) % (dilation * 4) :: dilation * 4]
    input_chunk = np.concatenate(
        [
            reshape_data(i, zone_size, window, shift)
            for i in [input_d0, input_d1, input_d2, input_d3]
        ],
        axis=0,
    )
    # input_chunk = np.squeeze(input_chunk, axis=0)
    return input_chunk


#####
def transform_input_chunk(sample_width, dilation, zone_size, window, shift, batch):
    input_d0 = batch["input"]
    target = batch["target"]
    input_chunk = make_input_chunk(
        input_d0,
        sample_width,
        dilation,
        zone_size,
        window,
        shift,
    )
    target_chunk = make_input_chunk(
        target[:-1],
        sample_width,
        dilation,
        zone_size,
        window,
        shift,
    )
    target = np.array(target[1:])
    return dict(input=input_chunk, to_interleave=target_chunk, target=target)


def mix_input_and_output(batch):
    input = batch["input"]
    to_interleave = batch["to_interleave"]
    target = batch["target"]
    input_chunk = Paul(np.array(input), np.array(to_interleave))
    return dict(input=input_chunk, target=target)


#####


def audio_gen(pair, window, stride, normalize=True):
    def _audio_gen():
        i, _ = librosa.load(pair[0], sr=44100)
        o, _ = librosa.load(pair[1], sr=44100)
        start = 0
        normy = librosa.util.normalize if normalize else lambda x: x
        while start + window <= len(i):
            for m in [1.0, -1.0]:
                ii = i[start : start + window] * m
                oo = o[start : start + window]
                yield {
                    "input": normy(ii),
                    "target": normy(oo),
                }
            start += stride

    return _audio_gen


def Paul(a, b):
    assert a.shape[-1] == b.shape[-1]
    c = np.empty(
        (
            *a.shape[:-1],
            a.shape[-1] + b.shape[-1],
        ),
        dtype=a.dtype,
    )
    c[..., 0::2] = a
    c[..., 1::2] = b
    return c


def audio_gen_2d(pair, window, stride, normalize=True):
    def _audio_gen():
        i, _ = librosa.load(pair[0], sr=44100)
        o, _ = librosa.load(pair[1], sr=44100)
        start = 0
        normy = librosa.util.normalize if normalize else lambda x: x
        while start + window + 1 <= len(i):
            for m in [1.0, -1.0]:
                ii = i[start + 1 : start + 1 + window]
                oo = o[start : start + 1 + window] * m
                yield {
                    "input": normy(ii),
                    "target": normy(oo),
                }
            start += stride

    return _audio_gen


def get_total_len(path, window, stride):
    with wave.open(path, "rb") as wav_file:
        num_samples = wav_file.getnframes()
        return (num_samples - window) // stride


def get_total_len_2d(path, window, stride):
    with wave.open(path, "rb") as wav_file:
        num_samples = wav_file.getnframes()
        return (num_samples - window - 1) // stride


def get_total_lens(paths, window, stride, f=get_total_len):
    return sum([f(x[0], window, stride) for x in paths], 0)


def make_data(paths, window, stride, feature_dim=-1, normalize=True):
    dataset = (
        interleave_datasets(
            [
                IterableDataset.from_generator(
                    audio_gen(pair, window, stride, normalize=normalize)
                )
                for pair in paths
            ]
        )
        .map(
            lambda x: {
                "input": np.expand_dims(x["input"], axis=feature_dim),
                "target": np.expand_dims(x["target"], axis=feature_dim),
            },
        )
        .shuffle(seed=42, buffer_size=2**10)
        # .with_format("jax")
    )

    return dataset, get_total_lens(paths, window, stride) * 2


def make_2d_data(paths, window, stride, feature_dim=-1, shuffle=True, normalize=True):
    dataset = (
        interleave_datasets(
            [
                IterableDataset.from_generator(
                    audio_gen_2d(pair, window, stride, normalize=normalize)
                )
                for pair in paths
            ]
        )
        .map(
            lambda x: {"to_interleave": x["target"][:-1], **x},
        )
        .map(mix_input_and_output, remove_columns=["to_interleave"])
        .map(partial(truncate_target, window))
        .map(
            lambda x: {
                "input": np.expand_dims(x["input"], axis=feature_dim),
                "target": np.expand_dims(x["target"], axis=feature_dim),
            },
        )
    )
    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=2**10)
    dataset = dataset  # .with_format("jax")

    # * 2 because we do flip for data augmentation
    return dataset, get_total_lens(paths, window, stride, f=get_total_len_2d) * 2


def truncate_target(window, batch):
    input = batch["input"]
    target = batch["target"]
    target = target[-window:]
    return dict(input=input, target=target)


def make_2d_data_with_delays_and_dilations(
    paths,
    window,
    stride,
    shift,
    dilation,
    channels,
    feature_dim=-1,
    shuffle=True,
    normalize=True,
):
    zone_size = channels // 4
    sample_width = (window + (zone_size * shift)) * (4 * dilation)
    dataset = (
        interleave_datasets(
            [
                IterableDataset.from_generator(
                    audio_gen_2d(pair, sample_width, stride, normalize=normalize)
                )
                for pair in paths
            ]
        )
        .map(
            partial(
                transform_input_chunk, sample_width, dilation, zone_size, window, shift
            )
        )
        .map(mix_input_and_output, remove_columns=["to_interleave"])
        .map(partial(truncate_target, window))
        .map(
            lambda x: {
                "input": np.transpose(x["input"], (1, 0)),
                "target": np.expand_dims(x["target"], axis=feature_dim),
            },
        )
    )
    if shuffle:
        dataset = dataset.shuffle(seed=42, buffer_size=2**10)
    dataset = dataset  # .with_format("jax")

    # * 2 because we do flip for data augmentation
    return dataset, get_total_lens(paths, window, stride, f=get_total_len_2d) * 2


if __name__ == "__main__":
    from get_files import FILES

    dataset, _ = make_2d_data(
        FILES[:1], 2**16, 2**8
    )  # paths, window, stride, shift, dilation, channels
    # dataset, _ = make_2d_data_with_delays_and_dilations(
    #     paths=FILES[:1],
    #     window=2**15,
    #     stride=2**8,
    #     shift=16,
    #     dilation=1,
    #     channels=2**3,
    #     feature_dim=-1,
    # )
    batch = next(dataset.iter(8, drop_last_batch=True))
    i = np.array(batch["input"])
    o = np.array(batch["target"])
    assert i.shape[1] == o.shape[1] * 2
    soundfile.write("/tmp/ipt.wav", i[0], 44100)
    soundfile.write("/tmp/opt.wav", o[0], 44100)
