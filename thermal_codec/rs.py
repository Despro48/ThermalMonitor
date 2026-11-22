"""Reed-Solomon FEC for profile B: shortened RS(222,190) over GF(256).

Primitive polynomial 0x11D, generator alpha = 2, 32 parity bytes, which
corrects up to 16 corrupted bytes. This is RS(255,223) shortened by 33 leading
zero symbols; encoding/decoding operates directly on 222 symbols.
"""

from __future__ import annotations

PRIMITIVE_POLY = 0x11D
GENERATOR = 2
NSYM = 32
DATA_BYTES = 190
PARITY_BYTES = NSYM
CODEWORD_BYTES = DATA_BYTES + PARITY_BYTES

_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= PRIMITIVE_POLY
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF(256) division by zero")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]


def gf_inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("GF(256) inverse of zero")
    return _EXP[(255 - _LOG[a]) % 255]


def gf_pow(a: int, power: int) -> int:
    return 0 if a == 0 else _EXP[(_LOG[a] * power) % 255]


def _poly_mul(p: list[int], q: list[int]) -> list[int]:
    result = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            log_a = _LOG[a]
            for j, b in enumerate(q):
                if b:
                    result[i + j] ^= _EXP[log_a + _LOG[b]]
    return result


def _poly_add(p: list[int], q: list[int]) -> list[int]:
    length = max(len(p), len(q))
    result = [0] * length
    for i in range(length):
        result[i] = (p[i] if i < len(p) else 0) ^ (q[i] if i < len(q) else 0)
    return result


def _poly_eval_high(p: list[int], x: int) -> int:
    value = 0
    for coefficient in p:
        value = gf_mul(value, x) ^ coefficient
    return value


def _poly_eval_low(p: list[int], x: int) -> int:
    value = 0
    for coefficient in reversed(p):
        value = gf_mul(value, x) ^ coefficient
    return value


def _generator_polynomial() -> list[int]:
    generator = [1]
    for i in range(NSYM):
        generator = _poly_mul(generator, [1, gf_pow(GENERATOR, i)])
    return generator


_GENERATOR_POLY = _generator_polynomial()


def rs_encode(data: bytes) -> bytes:
    """Encode 190 data bytes into a 222-byte systematic codeword."""
    if len(data) != DATA_BYTES:
        raise ValueError(f"expected {DATA_BYTES} data bytes, got {len(data)}")
    work = list(data) + [0] * NSYM
    for i in range(DATA_BYTES):
        coefficient = work[i]
        if coefficient:
            log_coefficient = _LOG[coefficient]
            for j in range(1, len(_GENERATOR_POLY)):
                work[i + j] ^= _EXP[log_coefficient + _LOG[_GENERATOR_POLY[j]]]
    return bytes(data) + bytes(work[DATA_BYTES:])


def _berlekamp_massey(syndromes: list[int]) -> list[int]:
    """Return the error locator polynomial (ascending coefficients)."""
    locator = [1]
    previous = [1]
    length = 0
    shift = 1
    previous_discrepancy = 1
    for n in range(NSYM):
        discrepancy = syndromes[n]
        for i in range(1, length + 1):
            discrepancy ^= gf_mul(locator[i], syndromes[n - i])
        if discrepancy == 0:
            shift += 1
            continue
        coefficient = gf_mul(discrepancy, gf_inv(previous_discrepancy))
        shifted = [0] * shift + [gf_mul(coefficient, value) for value in previous]
        if 2 * length <= n:
            saved = locator[:]
            locator = _poly_add(locator, shifted)
            length = n + 1 - length
            previous = saved
            previous_discrepancy = discrepancy
            shift = 1
        else:
            locator = _poly_add(locator, shifted)
            shift += 1
    return locator[: length + 1]


def rs_decode(codeword: bytes) -> tuple[bytes, int]:
    """Correct a 222-byte codeword. Returns (codeword, corrected_symbols).

    Raises ValueError if the word is uncorrectable.
    """
    if len(codeword) != CODEWORD_BYTES:
        raise ValueError(f"expected {CODEWORD_BYTES} bytes, got {len(codeword)}")

    syndromes = [_poly_eval_high(list(codeword), gf_pow(GENERATOR, i)) for i in range(NSYM)]
    if not any(syndromes):
        return bytes(codeword), 0

    locator = _berlekamp_massey(syndromes)
    degree = len(locator) - 1
    if degree == 0 or 2 * degree > NSYM:
        raise ValueError("uncorrectable Reed-Solomon block")

    positions: list[int] = []  # positions counted from the last symbol (0-based)
    for position in range(CODEWORD_BYTES):
        inverse_locator = gf_pow(GENERATOR, (-position) % 255)
        if _poly_eval_low(locator, inverse_locator) == 0:
            positions.append(position)
    if len(positions) != degree:
        raise ValueError(f"Chien search found {len(positions)} roots, expected {degree}")

    omega = _poly_mul(syndromes, locator)[:NSYM]
    derivative = [0] * max(1, len(locator) - 1)
    for j in range(len(derivative)):
        i = j + 1
        derivative[j] = locator[i] if i < len(locator) and (i & 1) else 0

    corrected = bytearray(codeword)
    for position in positions:
        x = gf_pow(GENERATOR, position)
        inverse_x = gf_pow(GENERATOR, (-position) % 255)
        denominator = _poly_eval_low(derivative, inverse_x)
        if denominator == 0:
            raise ValueError("zero Forney denominator")
        numerator = _poly_eval_low(omega, inverse_x)
        magnitude = gf_mul(x, gf_div(numerator, denominator))
        corrected[CODEWORD_BYTES - 1 - position] ^= magnitude

    if any(_poly_eval_high(list(corrected), gf_pow(GENERATOR, i)) for i in range(NSYM)):
        raise ValueError("Reed-Solomon miscorrection")
    return bytes(corrected), len(positions)


__all__ = [
    "CODEWORD_BYTES",
    "DATA_BYTES",
    "GENERATOR",
    "NSYM",
    "PARITY_BYTES",
    "gf_div",
    "gf_inv",
    "gf_mul",
    "gf_pow",
    "rs_decode",
    "rs_encode",
]
