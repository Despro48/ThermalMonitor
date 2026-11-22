# Thermal Image Compression over LoRa — Protocol Specification

Status: implementation spec for the Despro48 thermal link (32×24 MLX90640 → ESP32 LoRa → ESP32 receiver → USB serial → viewer).

This document defines two interchangeable compression profiles:

| Profile | Name | Delivery | Nominal size | Quality |
|---|---|---|---|---|
| **A** | Lossless MED + Golomb-Rice strips | 3 LoRa packets / frame | ~400–460 B/frame | bit-exact |
| **B** | Lossy 4×4 integer DCT + RS FEC | 1 LoRa packet / frame | 222 B/frame | ~0.1–0.25 °C RMS (typical scene) |

Both profiles share the same 32×24 frame geometry and 0.05 °C quantization step. The sender
chooses one profile; receiver and viewer are built per profile (the two implementations live on
separate branches).

---

## 1. System topology

```
MLX90640 (32×24 float °C)
        │ I²C
        ▼
Firmware ESP32 ── encoder ── LoRa 433 MHz ──► Receiver ESP32 ── USB serial ──► ThermalMonitor
                                                                              (GroundStation later)
```

* **Firmware** owns the encoder and the LoRa transmitter.
* **Receiver** is a transparent bridge: it validates the LoRa packet, then forwards the complete
  packet over USB serial inside a length-prefixed record. It does **not** decode images, so the
  ESP32 RX firmware stays small and the reference decoder lives where it can be iterated quickly.
* **ThermalMonitor** is the reference decoder and display. It decodes the forwarded packet,
  renders the frame, and is the place where new profiles are prototyped and tested.
* **GroundStation** will later consume the same serial records with a C++ port of the decoder.
  It is explicitly out of scope for the first implementation.

### 1.1 Common frame model

* Geometry: width `W = 32`, height `H = 24`, `PIXELS = 768`.
* Pixel order: row-major, origin top-left.
* Temperatures arrive as `float` °C from `Adafruit_MLX90640::getFrame()`.
* Quantization: `q = clamp(round_half_away_from_zero(T × 20), -32768, 32767)`.
  * LSB = 0.05 °C.
  * Range = -1638.40 °C … +1637.35 °C (far beyond the sensor range).
  * `round_half_away_from_zero(x) = floor(x + 0.5)` for `x ≥ 0`, `ceil(x − 0.5)` otherwise.
  * All encoders and decoders must use this exact rule so C++ and Python agree.
* Decoded temperatures are `q × 0.05` °C.
* Non-finite input (NaN/Inf) is mapped to `q = 0` (0 °C) before encoding so a bad
  sensor sample cannot poison a whole frame.

---

## 2. Profile A — Lossless multi-packet strips

### 2.1 Pipeline

```
float frame
  → int16 q (0.05 °C LSB)
  → split into 3 strips of 32×8
  → per strip: 2D MED predictor
  → signed residual → zigzag (sign folding)
  → adaptive Golomb-Rice (k ∈ {1,2,3}) with 16-bit escape
  → 4-byte strip header + bitstream
  → 3 independent LoRa packets (normally; up to 12 with split fallback)
```

Predictor state is reset at every strip. A lost packet therefore damages exactly one strip
(8 rows); the other two strips remain bit-exact.

### 2.2 Strip layout and header

Strip `s ∈ {0,1,2}` contains rows `s×8 … s×8+7`, all 32 columns, scanned row-major inside the
strip. The first pixel of a strip, `p(0,0)`, is stored verbatim in the header.

Packet layout (4-byte header followed by the Rice bitstream):

| Offset | Field | Meaning |
|---|---|---|
| 0 | `frame_id` | 0…255, incremented once per captured frame, wraps |
| 1 | `strip_id \| (k << 2) \| (part << 4) \| (split << 6)` | bits 0–1: strip id 0…2; bits 2–3: Rice parameter `k` (1…3; 0 = raw split part); bits 4–5: split part 0…3; bit 6: split flag; bit 7: reserved 0 |
| 2 | `ref_hi` | high byte of `ref` (int16, two's complement) |
| 3 | `ref_lo` | low byte of `ref` |
| 4… | payload | Golomb-Rice bitstream, MSB-first, zero-padded to a byte boundary; or raw int16 pixels for split `k=0` |

`ref = p(0,0)` is **not** coded as a residual. The decoder writes `p(0,0) = ref` and starts
decoding at `x=1, y=0`.

Maximum LoRa payload is 255 bytes, so the nominal Rice payload must be ≤ 251 bytes. Strips that
exceed it use the overflow split fallback in §2.7 rather than being dropped.

### 2.3 MED predictor (LOCO-I)

For each pixel other than `(0,0)`, with `A = left`, `B = above`, `C = above-left`:

```
pred = median(A, B, A + B − C)
```

Boundary rules (must match exactly in encoder and decoder):

* `y == 0, x > 0`: `A = p(x−1,0)`, `B = A`, `C = A` → `pred = A`.
* `x == 0, y > 0`: `B = p(0,y−1)`, `A = B`, `C = B` → `pred = B`.
* otherwise: normal `A`, `B`, `C`.

`pred` is clamped to `[-32768, 32767]`. The residual `r = p(x,y) − pred` is a signed 32-bit
value. (It can exceed the int16 range for arbitrary input; the escape path below keeps the codec
lossless for any int16 frame.)

### 2.4 Zigzag and Rice coding

Sign folding (`uint32`):

```
u = (r >= 0) ? 2*r : -2*r - 1
```

For a strip, the encoder chooses `k ∈ {1,2,3}` that minimizes the exact encoded bit count.
The decoder reads `k` from the header.

For each residual, with `q = u >> k`:

* Normal code (`q < 15`): write `q` one-bits, one zero-bit, then the low `k` bits of `u`
  (MSB first). Length = `q + 1 + k`.
* Escape code (`q ≥ 15`): write fifteen one-bits, one zero-bit, then **the raw int16 pixel
  value** (two's complement, MSB first, 16 bits). This guarantees losslessness even when the
  residual does not fit in 16 bits.
* The decoder counts one-bits until a zero. Count 15 means escape: read 16 bits as the absolute
  pixel value. Count 0…14 means normal: read `k` bits, reconstruct
  `u = (count << k) | remainder`, then `r = (u >> 1) ^ -(u & 1)`.

The bitstream is written MSB-first. After the last of the 255 coded pixels, pad the final byte
with zero bits; the decoder knows the pixel count and ignores padding.

### 2.5 Size expectations

Measured on synthetic 32×24 scenes (gradient + Gaussian blobs):

| Sensor noise (σ) | Total (3 packets) | Per-strip payload |
|---|---|---|
| 0.02 °C | 403–409 B | 95–101 B |
| 0.05 °C | 417–422 B | 99–105 B |
| 0.08 °C | 432–443 B | 102–113 B |
| 0.10 °C | 445–458 B | 106–117 B |

Headers add 12 B/frame. Payloads are therefore ~1.5–2.9 % of the raw 1536 B float16 frame.
Large hot objects need more bits: a 150 °C Gaussian blob on a 22 °C background encodes to
999 B in 9 packets (still bit-exact); a full-range random frame uses the raw split fallback
(12 × 132 B).

### 2.6 Packet loss handling

Each strip (and each split part) is independently decodable, and the codec exposes per-packet
decode for partial rendering. The reference viewer currently waits for a complete frame
(3 nominal packets, or all four parts of a split strip) before publishing, so a lost packet
drops that frame. `frame_id` lets the viewer discard stale packets from an older frame while
reassembly is in progress.

### 2.7 Overflow split fallback (implementation extension)

A 32×8 strip can exceed the 251-byte payload budget on steep thermal edges. The nominal header
has spare bits, so the encoder falls back without changing normal frames:

1. Try the nominal 32×8 Rice packet. If payload ≤ 251 B, send one packet (`split = 0`).
2. Otherwise split the strip into four 32×2 parts (`split = 1`, `part = 0…3`) and try Rice on
   each part with its own `k` and reference pixel.
3. If a 32×2 part still exceeds 251 B (extremely noisy input), send it raw: `k = 0`, payload =
   64 little-endian int16 pixels (128 B), `ref = 0`. This always fits and is still lossless.
4. The decoder buffers parts per `(frame_id, strip_id)` and concatenates parts 0…3. A frame
   therefore uses 3 packets normally and up to 12 packets for pathological scenes.

The nominal 3-packet path and its header layout are unchanged; only scenes that would otherwise
be dropped use the extension.

---

## 3. Profile B — Lossy single packet with Reed–Solomon FEC

### 3.1 Pipeline

```
float frame
  → int16 q (0.05 °C LSB)
  → subtract frame minimum (int16) → uint16 residuals
  → 4×4 integer DCT (48 blocks)
  → quantization + rate loop (payload ≤ 186 B)
  → zigzag AC scan + DPCM DC + Exp-Golomb
  → 4-byte header + 186-byte payload = 190-byte data
  → shortened RS(222,190), 32 parity bytes
  → one 222-byte LoRa packet
```

### 3.2 Data layout before FEC

| Offset | Field | Meaning |
|---|---|---|
| 0 | `frame_id` | 0…255 |
| 1 | `min_lo` | low byte of `min_q` (int16, two's complement) |
| 2 | `min_hi` | high byte of `min_q` |
| 3 | `q_scale` | 1…255 = uniform quantizer step; 0 = DC-only fallback |
| 4…189 | payload | entropy bitstream, zero-padded to 186 bytes |

The 190-byte block is the RS data word. `frame_id`, `min_q`, and `q_scale` are protected by the
FEC exactly like the payload (a corrupted header is repaired by RS).

### 3.3 Forward 4×4 integer DCT

Input block `B` is a 4×4 matrix of `uint16` residuals (0…65535). Use the HEVC-style integer
basis:

```
M = [ 64  64  64  64 ]
    [ 83  36 -36 -83 ]
    [ 64 -64 -64  64 ]
    [ 36 -83  83 -36 ]
```

All arithmetic is `int64`:

```
C = (M · B · Mᵀ + 32768) >> 16          // forward
B' = (Mᵀ · C · M + 2048) >> 12          // inverse
```

Properties: a constant block of value `v` produces `C[0][0] = v` and all other coefficients 0;
the inverse returns `v`. `MᵀM ≈ 16384·I`, which is where the `>>16` / `>>12` scaling comes from.

Block order: raster order over 8×6 blocks, each block row-major over 4×4 pixels.

### 3.4 Quantization and rate loop

Uniform scalar quantizer with step `q_scale`:

```
qc = sign(C) · ((|C| + q_scale/2) / q_scale)     // integer division
C' = qc · q_scale                                  // decoder
```

Encoder rate loop (binary search over `q_scale = 1…255`):

1. DCT the frame once (coefficients are reused).
2. Quantize + entropy-encode for the candidate step.
3. If the byte-padded bitstream is ≤ 186 B, try a smaller step; otherwise try a larger one.
4. The smallest step that fits is used. This yields the best quality within the fixed 190-byte
   data budget.

If even `q_scale = 255` cannot fit (extremely noisy/adversarial input), use the **DC-only
fallback** (`q_scale = 0`): payload = 48 block means as little-endian `uint16`, zero-padded to
186 B. This always fits and produces a blocky but valid image. Normal thermal scenes never hit
this path.

### 3.5 Entropy coding

Bitstream is MSB-first, byte-padded with zeros. For each of the 48 blocks in raster order:

1. **DC**: DPCM against the previous block's quantized DC (`prev_dc`, initially 0).
   `delta = dc − prev_dc`. Map to unsigned `u = (delta ≥ 0) ? 2·delta : −2·delta − 1`
   and write order-0 Exp-Golomb `u`:
   `floor(log2(u+1))` zero bits followed by `u+1` in binary.
2. **AC**: the 15 AC coefficients in 4×4 zigzag order (positions 1…15):
   ```
   zigzag = [(0,0),(1,0),(0,1),(0,2),(1,1),(2,0),(3,0),(2,1),
             (1,2),(0,3),(1,3),(2,2),(3,1),(3,2),(2,3),(3,3)]
   ```
   * If all 15 are zero: write a single `0` bit.
   * Otherwise: write `1`, then the 15-bit significance bitmap (bit 14 = position 1,
     bit 0 = position 15), then for each set bit in scan order: sign bit (`1` = negative) and
     Exp-Golomb of `|level| − 1` (unsigned, order-0).

The decoder knows there are exactly 48 blocks, so no end-of-frame marker is needed. Bytes after
the last decoded bit are padding.

### 3.6 Reed–Solomon FEC

Shortened **RS(222,190)** over GF(256):

* Primitive polynomial `0x11D`, generator `α = 2`, 32 parity bytes.
* This is RS(255,223) shortened by 33 leading zero symbols; encode/decode operates directly on
  222 symbols.
* Corrects up to **16 corrupted bytes** (any positions, including the 4-byte header).
* On uncorrectable input the decoder **drops the packet** (no partial frame).
* The receiver forwards the full 222 bytes; FEC correction happens in the viewer.

Optional SF9 / narrow-MTU variant (not built by default): **RS(115,99)**, 16 parity bytes,
4-byte header + 95-byte payload, corrects 8 bytes.

### 3.7 Size and quality expectations

* Packet size is always exactly 222 B (4 header + 186 payload + 32 parity).
* Synthetic scenes (gradient + blobs, 0.08 °C noise): `q_scale` 4–10, RMS ≈ 0.17–0.19 °C,
  max error ≈ 0.55–0.70 °C.
* Typical thermal scenes with ≤ 60 °C span should stay at RMS ≤ 0.25 °C. A 160 °C span
  (e.g. −40…+120 °C) measured ≈ 0.38 °C at the forced `q_scale`.
* At SF7 / BW 500 kHz / CR 4/6 a 222-byte packet occupies one LoRa frame; at SF9 / BW 125 kHz
  use the 115-byte variant and expect lower frame rate.

---

## 4. Receiver → viewer serial protocol

The receiver never emits CSV temperatures in these profiles. It emits length-prefixed binary
records so the viewer sees exactly what the radio delivered (including FEC parity, for profile B).

### 4.1 Profile A record

```
"THA,<len>:"  followed by <len> raw bytes  (the complete LoRa packet: 4-byte header + payload)
```

`<len>` is decimal ASCII, 4…255. The viewer validates `strip_id` and `k`, buffers strips by
`frame_id`, and publishes a frame when all three strips have arrived.

### 4.2 Profile B record

```
"THB,<len>:"  followed by <len> raw bytes  (the complete 222-byte LoRa packet)
```

`<len>` is decimal ASCII, normally 222. The viewer runs RS correction, then entropy/IDCT decode.

### 4.3 Framing rules

* Header is ASCII; payload is arbitrary binary, including `,`, `:`, `\n`, and 0x00.
* The parser scans for `THA,` / `THB,`, parses the decimal length up to the next `:`, then reads
  exactly that many bytes. Bytes outside a record (startup banners, debug lines) are discarded.
* Maximum record payload is 255 bytes; a declared length above that is treated as a parse error
  and the parser resynchronizes.
* Records are forwarded immediately (no frame buffering on the ESP32).

---

## 5. Dummy-data test plan and acceptance criteria

All tests are deterministic and run on the host; no hardware is required.

1. **Dummy scene generator** — 32×24 float frame: linear gradient, four Gaussian hot/cold
   blobs, Gaussian sensor noise with a configurable σ (default 0.08 °C), seeded RNG.
2. **Profile A round trip** — quantize → 3 strips → encode → decode → compare int16 arrays.
   * Must be bit-exact for every seed and for adversarial int16 frames (including escapes).
   * Packet size ≤ 255 B for realistic scenes; total ≤ 460 B at σ = 0.1 °C.
   * A 150 °C blob and a full-range random frame must use the split fallback and still decode
     bit-exactly with every packet ≤ 255 B.
3. **Profile A packet loss** — drop strip 1; strips 0 and 2 must still decode exactly. A missing
   split part must only cost its 2 rows.
4. **Profile B round trip** — encode → RS → inject 0…16 random byte errors → RS decode →
   entropy/IDCT decode. Must match the no-error decode exactly and report the corrected count.
5. **Profile B rate bound** — encoded data is always exactly 190 B before FEC and 222 B after;
   payload never exceeds 186 B. Random/adversarial frames must take the DC-only fallback and
   still fit.
6. **Profile B quality** — on the default synthetic scene, RMS error ≤ 0.25 °C and max error
   ≤ 1.0 °C.
7. **Cross-language check** — Python-generated `int16` fixture encoded by the C++ firmware
   encoder and decoded by the Python reference decoder (and the reverse ordering where useful)
   must agree bit-exactly (A) / within the same quantized output (B).
8. **Serial framing** — synthetic packets fed through the receiver formatter and parsed back by
   the viewer parser must preserve every byte and reject malformed lengths.

Commands (from the repository root; the file suffixes match the branch):

```bash
# ThermalMonitor (both branches)
uv sync
uv run python -m unittest discover -s tests -v

# Profile A branch
uv run python tools/simulate_a.py
g++ -std=c++17 -O2 -Wall -Wextra src/thermal_codec_a.cpp test/host/test_option_a.cpp -o /tmp/test_a && /tmp/test_a
g++ -std=c++17 -O2 -Wall -Wextra src/serial_forward_a.cpp test/host/test_receiver_a.cpp -o /tmp/test_rx_a && /tmp/test_rx_a

# Profile B branch
uv run python tools/simulate_b.py --corrupt
g++ -std=c++17 -O2 -Wall -Wextra src/thermal_codec_b.cpp test/host/test_option_b.cpp -o /tmp/test_b && /tmp/test_b
g++ -std=c++17 -O2 -Wall -Wextra src/serial_forward_b.cpp test/host/test_receiver_b.cpp -o /tmp/test_rx_b && /tmp/test_rx_b

# Firmware (ESP32) build, either branch
pio run
```

Cross-language checks use a fixture written by `tools/simulate_{a,b}.py --fixture ...`, the
firmware host test's `--fixture/--packets` (or `--packet`) options, and
`tools/verify_packets_{a,b}.py`.

---

## 6. Selection rule

* **Profile A** — use when exact radiometry is required (measurement, logging, later analysis)
  and ~1.5–3 FPS is acceptable. No FEC; a lost packet costs 8 rows of one frame.
* **Profile B** — use for live search/navigation where 2.5–5.5 FPS and single-packet delivery
  matter, and where 0.1–0.25 °C RMS is within sensor noise. RS(222,190) absorbs up to 16
  corrupted bytes per packet.
* The two profiles are not interoperable on the same receiver without a mode switch. Build one
  branch per profile, or add a one-byte profile tag to the serial record if both must coexist.

---

## 7. Out of scope / next steps

* GroundStation parser + C++ decoder port for `THA` / `THB` records.
* Optional SF9 MTU profile (RS(115,99)).
* Optional frame-level interleaving / repeat for profile A packet loss.
* On-hardware airtime measurement and FPS validation with the MLX90640 at 8 Hz.
