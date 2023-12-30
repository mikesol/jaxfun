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
                "input": i[start : start + window],
                "target": i[start : start + window],
                "input_path": pair[0],
                "target_path": pair[1],
                "start": start,
            }
            start += stride

    return _audio_gen

def get_total_len(path, window, stride):
    with wave.open(path, 'rb') as wav_file:
        num_samples = wav_file.getnframes()
        return (num_samples - window) // stride

def get_total_lens(paths, window, stride):
    return sum([get_total_len(x[0], window, stride) for x in paths], 0)

def make_data(paths, window, stride):
    dataset = (
        interleave_datasets(
            [
                IterableDataset.from_generator(audio_gen(pair, window, stride))
                for pair in paths
            ]
        )
        .map(
            lambda x: {
                "input": np.expand_dims(x["input"], axis=-1),
                "target": np.expand_dims(x["target"], axis=-1),
            }, remove_columns=["input_path", "target_path", "start"]
        )
        .shuffle(seed=42, buffer_size=2**10)
        .with_format("jax")
    )

    return dataset, get_total_lens(paths, window, stride)


if __name__ == "__main__":
    from get_files import FILES
    dataset = make_data(FILES[:1], 2**16, 2**8)
    print(next(dataset.iter(8, drop_last_batch=True))["input"].shape)
