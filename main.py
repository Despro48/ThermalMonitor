import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Serial port configuration
serial_port = 'COM19'  # Replace with your serial port
baud_rate = 921600
IMAGE_SCALE = 2
IMAGE_HEIGHT = 24
IMAGE_WIDTH = 32
PIXEL_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH


# Initialize serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)
ser.flush()


# Initialize temperature array
temps = np.zeros(PIXEL_COUNT)
serial_buffer = bytearray()


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
    if ser.in_waiting:
        serial_buffer.extend(ser.read(ser.in_waiting))

    while b'\n' in serial_buffer:
        line_end = serial_buffer.index(b'\n')
        line = bytes(serial_buffer[:line_end]).strip()
        del serial_buffer[:line_end + 1]

        try:
            values = np.asarray(line.decode('ascii').split(','), dtype=float)
        except (UnicodeDecodeError, ValueError):
            continue

        if values.size != PIXEL_COUNT or not np.isfinite(values).all():
            continue

        min_temp = values.min()
        max_temp = values.max()
        if min_temp == max_temp:
            temps[:] = 260
        else:
            temps[:] = np.interp(values, [min_temp, max_temp], [160, 360])


def update_heatmap(*args):
    read_serial_data()
    image = temps.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    heatmap.set_array(bilinear_interpolate(image, IMAGE_SCALE))
    return heatmap,


ani = animation.FuncAnimation(fig, update_heatmap, interval=10, cache_frame_data=False)

plt.show()