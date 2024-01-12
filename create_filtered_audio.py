import numpy as np
from scipy import signal
# from concurrent.futures import ProcessPoolExecutor
from scipy.signal import lfilter

# EXECUTOR = ProcessPoolExecutor()

def apply_filter(params):
    audio, fs, f0, Q = params
    b, a = signal.iirpeak(f0, Q, fs)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def create_filtered_audio(audio, num_filters, fs, f0_start, f0_end, Qstart, Qend):
    f0_values = np.logspace(np.log10(f0_start), np.log10(f0_end), num_filters)
    Q_values = np.logspace(np.log10(Qstart), np.log10(Qend), num_filters)
    filtered_audios = [apply_filter(audio, fs, f0, Q) for f0, Q in zip(f0_values, Q_values)]

    # Stack the filtered audios to create a (n, num_filters) array
    filtered_audio_stack = np.concatenate(filtered_audios, axis=-1)

    return filtered_audio_stack