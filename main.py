import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Serial port configuration
serial_port = 'COM10'  # Replace with your serial port
baud_rate = 921600
IMAGE_SCALE = 2
IMAGE_HEIGHT = 24
IMAGE_WIDTH = 32


# Initialize serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)
ser.flush()


# Initialize temperature array
temps = np.zeros(IMAGE_HEIGHT * IMAGE_WIDTH)
max_temp = 0
min_temp = 500


# Create the figure for plotting
fig, ax = plt.subplots()
heatmap = ax.imshow(
    np.zeros((IMAGE_HEIGHT * IMAGE_SCALE, IMAGE_WIDTH * IMAGE_SCALE)),
    cmap='hsv_r',
    vmin=160,
    vmax=360,
    interpolation='nearest',
)
plt.colorbar(heatmap)


def bilinear_interpolate(image, scale):
    if scale == 1:
        return image

    height, width = image.shape
    output_height = height * scale
    output_width = width * scale

    y = np.linspace(0, height - 1, output_height)
    x = np.linspace(0, width - 1, output_width)
    x_grid, y_grid = np.meshgrid(x, y)

    x0 = np.floor(x_grid).astype(int)
    y0 = np.floor(y_grid).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    x_weight = x_grid - x0
    y_weight = y_grid - y0

    top = image[y0, x0] * (1 - x_weight) + image[y0, x1] * x_weight
    bottom = image[y1, x0] * (1 - x_weight) + image[y1, x1] * x_weight
    return top * (1 - y_weight) + bottom * y_weight


def read_serial_data():
    global max_temp, min_temp
    if ser.in_waiting > 5000:
        line = ser.read_until(b'\r')
        if len(line) > 4608:
            line = line[:4608]
        split_string = line.decode().strip().split(',')

        # Update min and max temperatures
        max_temp = 0
        min_temp = 50

        for q in range(768):
            try:
                value = float(split_string[q])
                if value > max_temp:
                    max_temp = value
                if value < min_temp:
                    min_temp = value
            except (ValueError, IndexError):
                pass
        
        # Map temperatures to colors
        for q in range(768):
            try:
                value = float(split_string[q])
                mapped_value = np.clip(np.interp(value, [min_temp, max_temp], [160, 360]), 160, 360)
                temps[q] = mapped_value
            except (ValueError, IndexError):
                temps[q] = 0


def update_heatmap(*args):
    read_serial_data()
    image = temps.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    heatmap.set_array(bilinear_interpolate(image, IMAGE_SCALE))
    return heatmap,


ani = animation.FuncAnimation(fig, update_heatmap, interval=10, cache_frame_data=False)

plt.show()