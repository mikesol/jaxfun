import jax.numpy as jnp
from jax import vmap

def logarithmic_fade_in(seq_length):
    # Create a logarithmic curve, which starts at a very small value to avoid log(0)
    log_curve = jnp.log(jnp.linspace(1e-10, 1, seq_length))
    # Normalize the curve to have values from 0 to 1
    normalized_curve = (log_curve - log_curve.min()) / (log_curve.max() - log_curve.min())
    return normalized_curve

def apply_fade_in(audio_data):
    # Assuming audio_data shape is (batch, seq, chan)
    batch_size, seq_length, _ = audio_data.shape
    
    # Create the fade-in curve
    fade_in_curve = logarithmic_fade_in(seq_length)

    # Broadcast the fade-in curve to match the audio data shape
    fade_in_curve = fade_in_curve.reshape((1, seq_length, 1))
    # fade_in_curve = jnp.broadcast_to(fade_in_curve, audio_data.shape)

    # Apply the fade-in curve to the audio data
    faded_audio = audio_data * fade_in_curve
    return faded_audio

if __name__ == '__main__':
    audio_data = jnp.ones((1, 100, 1))
    faded_audio = apply_fade_in(audio_data)
    print(faded_audio.shape)
    print(faded_audio[0][0][0])
    print(faded_audio[0][99][0])
    print(faded_audio[0][50][0])
