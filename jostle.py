import numpy as np
import librosa
import soundfile as sf


def process_audio(audio_buffer, window_size, phase_jostle_deg):
    """
    Process an audio buffer with STFT, lightly jostle the phase, and resynthesize using ISTFT.

    :param audio_buffer: Input audio buffer.
    :param window_size: Window size for STFT.
    :param phase_jostle_deg: Max degree for phase offset in either direction.
    :return: Processed audio.
    """

    # STFT of the audio buffer
    stft_matrix = librosa.stft(audio_buffer, n_fft=window_size, window="hann")

    # Convert to magnitude and phase
    magnitude, phase = librosa.magphase(stft_matrix)

    # Jostle phase
    phase_jostle_rad = np.radians(phase_jostle_deg)
    random_phase_offset = np.random.uniform(
        -phase_jostle_rad, phase_jostle_rad, phase.shape
    )
    jostled_phase = phase * np.exp(1j * random_phase_offset)

    # Combine magnitude with jostled phase
    jostled_stft_matrix = magnitude * jostled_phase

    # ISTFT to resynthesize audio
    resynthesized_audio = librosa.istft(
        jostled_stft_matrix, window="hann", dtype=np.float32
    )

    return resynthesized_audio


# Example usage
# Load an audio file (replace 'your_audio_file.wav' with the path to your file)
audio_buffer, sr = sf.read("../data/day1/nt1_middle_far_mid_48_8.wav")
I = 2**17
O = 2**18
audio_buffer = audio_buffer[I : I + O]
# Process the audio
window_size = 1024  # Example window size
phase_jostle_deg = 10  # Example phase jostle in degrees
processed_audio = process_audio(audio_buffer, window_size, phase_jostle_deg)

# Save the processed audio (optional)
sf.write("/tmp/nt1_middle_close_mid_38_12296.wav", audio_buffer, sr)
sf.write("/tmp/jostle_nt1_middle_close_mid_38_12296.wav", processed_audio, sr)
