import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import soundfile as sf
import numpy as np
I = '/Users/mikesol/Downloads/pred0.wav'
O = '/Users/mikesol/Downloads/targ0.wav'
input_wave = sf.read(I)[0]
target_wave = sf.read(O)[0]
ml = min(input_wave.shape[0], target_wave.shape[0])
input_wave = input_wave[:ml]
target_wave = target_wave[:ml]
loss_array = (input_wave - target_wave) ** 2
overall_loss = 0.25

compute_loss = lambda _, __: loss_array
get_loss = lambda x: loss_array[int(x)]
# Assuming `input_wave`, `predicted_wave`, and `loss_array` are numpy arrays
# and `sample_rate` is the sampling rate of your audio files.

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Plot the input and target waveforms
input_line, = ax.plot(input_wave, label='Input Waveform')
target_line, = ax.plot(target_wave, label='Target Waveform')
plt.legend()

# Add overall loss text
overall_loss = compute_loss(input_wave, target_wave)
loss_text = ax.text(0.05, 0.95, f'Overall Loss: {overall_loss}', transform=ax.transAxes)

# Set the playhead initial position
playhead_pos = 0
playhead, = ax.plot([playhead_pos, playhead_pos], [np.min(input_wave), np.max(target_wave)], 
                    color='green', linewidth=2, label='Playhead')

# Function to update the figure when new time is selected
def update(val):
    playhead_pos = sample_slider.val
    playhead.set_xdata([playhead_pos, playhead_pos])
    current_loss = get_loss(playhead_pos)
    loss_text.set_text(f'Current Loss: {current_loss:.5f}')
    fig.canvas.draw_idle()

# Function to zoom at the playhead position
def zoom(factor, axis):
    if axis == 'x':
        left, right = ax.get_xlim()
        center = playhead.get_xdata()[0]
        ax.set_xlim([center - (center - left) / factor, center + (right - center) / factor])
    elif axis == 'y':
        bottom, top = ax.get_ylim()
        center = (bottom + top) / 2
        ax.set_ylim([center - (center - bottom) / factor, center + (top - center) / factor])
    fig.canvas.draw_idle()


# Function to move the playhead to clicked location
def on_click(event):
    if event.inaxes == ax:
        playhead_pos = int(event.xdata)
        playhead.set_xdata([playhead_pos, playhead_pos])
        
        # Update the slider value and loss text
        sample_slider.set_val(playhead_pos)
        current_loss = compute_loss(input_wave[playhead_pos], target_wave[playhead_pos])
        loss_text.set_text(f'Current Loss: {current_loss:.5f}')
        
        fig.canvas.draw_idle()

# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)

# Create the slider
axcolor = 'lightgoldenrodyellow'
axsamples = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
sample_slider = Slider(axsamples, 'Sample', 0, len(input_wave) - 1, valinit=0, valstep=1)

sample_slider.on_changed(update)


# Create zoom buttons
axzoom_in_x = plt.axes([0.81, 0.05, 0.1, 0.075])
axzoom_out_x = plt.axes([0.81, 0.15, 0.1, 0.075])
axzoom_in_y = plt.axes([0.7, 0.05, 0.1, 0.075])
axzoom_out_y = plt.axes([0.7, 0.15, 0.1, 0.075])

button_zoom_in_x = Button(axzoom_in_x, 'Zoom In X')
button_zoom_out_x = Button(axzoom_out_x, 'Zoom Out X')
button_zoom_in_y = Button(axzoom_in_y, 'Zoom In Y')
button_zoom_out_y = Button(axzoom_out_y, 'Zoom Out Y')

button_zoom_in_x.on_clicked(lambda event: zoom(2, 'x'))
button_zoom_out_x.on_clicked(lambda event: zoom(0.5, 'x'))
button_zoom_in_y.on_clicked(lambda event: zoom(2, 'y'))
button_zoom_out_y.on_clicked(lambda event: zoom(0.5, 'y'))

plt.show()