import numpy as np
from scipy import signal
from scipy.io import wavfile


def create_allpass_filter(alpha):
    """Create a first-order all-pass filter"""
    b = np.array([-alpha, 1])  # Numerator coefficients
    a = np.array([1, -alpha])  # Denominator coefficients
    return b, a


def process_audio_with_allpass_filter(audio, fs, alpha):
    # Ensure audio is in the correct format (stereo to mono, if necessary)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert to mono by averaging channels

    # Create the all-pass filter
    b, a = create_allpass_filter(alpha)

    # Apply the filter
    processed_audio = signal.lfilter(b, a, audio)

    return processed_audio, fs


if __name__ == "__main__":
    alpha = -0.25  # Change this value for different phase characteristics
    # Example usage
    input_file = "../data/day1/nt1_middle_far_mid_48_8.wav"
    output_file = f"/tmp/allpass_{alpha}_nt1_middle_close_mid_38_12296.wav"
    # Load the audio file
    I = 2**17
    O = 2**18
    fs, audio = wavfile.read(input_file)
    audio = audio[I : I + O]
    processed_audio, fs = process_audio_with_allpass_filter(audio, fs, alpha)
    wavfile.write(output_file, fs, processed_audio.astype(np.int16))
