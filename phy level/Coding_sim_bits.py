# ---------------------------------------------------------------------------
# 3. Кодирование: биты ↔ символы
# ---------------------------------------------------------------------------

import numpy as np

def bits_to_symbols(bits: np.ndarray, sf: int, gray: bool = True) -> np.ndarray:
    M = 2 ** sf
    n_sym = len(bits) // sf
    symbols = np.zeros(n_sym, dtype=np.int32)

    for i in range(n_sym):
        chunk = bits[i * sf: (i + 1) * sf]
        val = int("".join(map(str, chunk.tolist())), 2)

        if gray:
            val = val ^ (val >> 1)  # binary → Gray

        symbols[i] = val % M

    return symbols


def gray_to_binary(x: int) -> int:
    result = x
    while x > 0:
        x >>= 1
        result ^= x
    return result


def symbols_to_bits(symbols: np.ndarray, sf: int, gray: bool = True) -> np.ndarray:
    bits = []

    for sym in symbols:
        val = int(sym)

        if gray:
            val = gray_to_binary(val)

        bits.extend([(val >> (sf - 1 - i)) & 1 for i in range(sf)])

    return np.array(bits, dtype=np.uint8)