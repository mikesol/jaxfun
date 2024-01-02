from datasets import IterableDataset, interleave_datasets
import librosa
import numpy as np
import wave


def audio_gen(pair, window, stride):
    def _audio_gen():
        i, _ = librosa.load(pair[0])
        o, _ = librosa.load(pair[1])
        start = 0
        while start + window <= len(i):
            yield {
                "input":i[start : start + window],
                "target": o[start : start + window]
            }
            start += stride

    return _audio_gen


def Paul(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def audio_gen_2d(pair, window, stride):
    def _audio_gen():
        i, _ = librosa.load(pair[0])
        o, _ = librosa.load(pair[1])
        start = 0
        while start + window <= len(i):
            for m in [-1.0, 1.0]:
                ii = i[start + 1 : start + 1 + window]
                oo = o[start : start + 1 + window] * m
                ii = Paul(ii, oo[:-1])
                oo = oo[1:]
                yield {
                    "input": ii,
                    "target": oo,
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


def make_data(paths, window, stride, feature_dim=-1):
    dataset = (
        interleave_datasets(
            [
                IterableDataset.from_generator(
                    audio_gen(pair, window, stride)
                )
                for pair in paths
            ]
        )
        .map(
            lambda x: {
                "input": np.expand_dims(x["input"], axis=feature_dim),
                "target": np.expand_dims(x["target"], axis=feature_dim),
            },
        )        .shuffle(seed=42, buffer_size=2**10)
        .with_format("jax")
    )

    return dataset, get_total_lens(paths, window, stride)


def make_2d_data(paths, window, stride, feature_dim=-1):
    dataset = (
        interleave_datasets(
            [
                IterableDataset.from_generator(
                    audio_gen_2d(pair, window, stride)
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
        .with_format("jax")
    )

    # * 2 because we do flip for data augmentation
    return dataset, get_total_lens(paths, window, stride, f=get_total_len_2d) * 2


if __name__ == "__main__":
    from get_files import FILES

    dataset, _ = make_2d_data(FILES[:1], 2**16, 2**8)
    batch = next(dataset.iter(8, drop_last_batch=True))
    i = batch["input"]
    o = batch["target"]
    print(i.shape, o.shape)
    assert i.shape[2] * 2 == o.shape[1]
