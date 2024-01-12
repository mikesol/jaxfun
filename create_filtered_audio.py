import numpy as np
from scipy import signal
from scipy.signal import lfilter



def create_filtered_audio(num_filters, fs, f0_start, f0_end, Qstart, Qend):
    f0_values = np.logspace(np.log10(f0_start), np.log10(f0_end), num_filters)
    Q_values = np.logspace(np.log10(Qstart), np.log10(Qend), num_filters)
    sigs = [signal.iirpeak(f0, Q, fs) for f0, Q in zip(f0_values, Q_values)]

    def _create_filtered_audio(audio):
        filtered_audios = [lfilter(b, a, audio) for b, a in sigs]

        # Stack the filtered audios to create a (n, num_filters) array
        filtered_audio_stack = np.concatenate(filtered_audios, axis=-1)

        return filtered_audio_stack

    return _create_filtered_audio
