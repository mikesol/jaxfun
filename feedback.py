import numpy as np
from scipy import signal
from scipy.io import wavfile


def process_audio(audio, c1, c2, c3):
    o = np.empty_like(audio)
    for i in range(len(audio)):
        if i == 0:
            o[i] = c1 * audio[i]
        elif i == 1:
            o[i] = c1 * audio[i] + c2 * o[i - 1]
        else:
            o[i] = c1 * audio[i] + c2 * o[i - 1] + c3 * o[i - 2]
    return o


if __name__ == "__main__":
    alpha = -0.25  # Change this value for different phase characteristics
    # Example usage
    input_file = "../data/day1/nt1_middle_far_mid_48_8.wav"
    output_file = f"/tmp/feedback_nt1_middle_close_mid_38_12296.wav"
    # Load the audio file
    I = 2**17
    O = 2**18
    fs, audio = wavfile.read(input_file)
    audio = audio[I : I + O]
    processed_audio = process_audio(audio, 0.7, -0.9, -0.35)
    wavfile.write(output_file, fs, processed_audio.astype(np.int16))
