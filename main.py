import argparse

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
PACKET_PAYLOAD_MAX = 253
SERIAL_PACKET_PREFIX = b'PKT,'
TEMP_MIN_C = 20.0
TEMP_MAX_C = 35.0
DYNAMIC_RANGE_SMOOTHING = 0.2
DYNAMIC_RANGE_PADDING = 0.05

parser = argparse.ArgumentParser()
parser.add_argument('--dynamic-min-max', action='store_true')
cli_args = parser.parse_args()


# Initialize serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)
ser.flush()


# Initialize temperature array
temps = np.full(PIXEL_COUNT, np.nan)
serial_buffer = bytearray()
packet_total = None
frame_buffer = bytearray()
packet_lengths = []
packet_seen = []


# Create the figure for plotting
fig, ax = plt.subplots()
heatmap = ax.imshow(
    np.full((IMAGE_HEIGHT * IMAGE_SCALE, IMAGE_WIDTH * IMAGE_SCALE), np.nan),
    cmap='hsv_r',
    vmin=TEMP_MIN_C,
    vmax=TEMP_MAX_C,
    interpolation='nearest',
)
plt.colorbar(heatmap).set_label('Temperature (°C)')

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
    global packet_total, frame_buffer, packet_lengths, packet_seen

    if ser.in_waiting:
        serial_buffer.extend(ser.read(ser.in_waiting))

    while True:
        marker = serial_buffer.find(SERIAL_PACKET_PREFIX)
        if marker < 0:
            del serial_buffer[:-len(SERIAL_PACKET_PREFIX) + 1]
            return
        if marker:
            del serial_buffer[:marker]

        header_end = serial_buffer.find(b':', len(SERIAL_PACKET_PREFIX))
        if header_end < 0:
            if len(serial_buffer) > 64:
                del serial_buffer[0]
                continue
            return

        try:
            idx, total, payload_len = map(
                int, serial_buffer[len(SERIAL_PACKET_PREFIX):header_end].split(b',')
            )
        except ValueError:
            del serial_buffer[0]
            continue

        if not (0 <= idx < total <= 255 and 0 < payload_len <= PACKET_PAYLOAD_MAX):
            del serial_buffer[0]
            continue

        packet_end = header_end + 1 + payload_len
        if len(serial_buffer) < packet_end:
            return

        payload = bytes(serial_buffer[header_end + 1:packet_end])
        del serial_buffer[:packet_end]
        # print(f"RX idx={idx}/{total} length={payload_len}", flush=True)

        if packet_total != total:
            packet_total = total
            frame_buffer = bytearray(total * PACKET_PAYLOAD_MAX)
            packet_lengths = [0] * total
            packet_seen = [False] * total

        offset = idx * PACKET_PAYLOAD_MAX
        frame_buffer[offset:offset + payload_len] = payload
        packet_lengths[idx] = payload_len
        packet_seen[idx] = True

        contiguous_len = 0
        for chunk_idx, seen in enumerate(packet_seen):
            if not seen:
                break
            chunk_len = packet_lengths[chunk_idx]
            if chunk_idx < total - 1 and chunk_len != PACKET_PAYLOAD_MAX:
                break
            contiguous_len = chunk_idx * PACKET_PAYLOAD_MAX + chunk_len

        if contiguous_len == 0:
            continue

        raw = bytes(frame_buffer[:contiguous_len])
        fields = raw.split(b',')
        if raw.endswith(b'\n'):
            fields[-1] = fields[-1].strip()
        else:
            fields.pop()

        values = []
        for field in fields:
            try:
                value = float(field)
            except ValueError:
                break
            if not np.isfinite(value):
                break
            values.append(value)

        if not values:
            continue

        values = np.asarray(values[:PIXEL_COUNT])
        temps[:values.size] = values


def update_heatmap(*args):
    global TEMP_MIN_C, TEMP_MAX_C

    read_serial_data()
    image = temps.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    if cli_args.dynamic_min_max:
        valid_temps = image[np.isfinite(image)]
        if valid_temps.size:
            target_min = float(np.min(valid_temps))
            target_max = float(np.max(valid_temps))
            padding = max((target_max - target_min) * DYNAMIC_RANGE_PADDING, 0.25)
            target_min -= padding
            target_max += padding
            TEMP_MIN_C += DYNAMIC_RANGE_SMOOTHING * (target_min - TEMP_MIN_C)
            TEMP_MAX_C += DYNAMIC_RANGE_SMOOTHING * (target_max - TEMP_MAX_C)
            heatmap.set_clim(TEMP_MIN_C, TEMP_MAX_C)

    heatmap.set_array(bilinear_interpolate(image, IMAGE_SCALE))
    return heatmap,


ani = animation.FuncAnimation(fig, update_heatmap, interval=10, cache_frame_data=False)

plt.show()