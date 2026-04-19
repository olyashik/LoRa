# ---------------------------------------------------------------------------
# 3. Кодирование Хэмминга (FEC / Coding Rate)
# ---------------------------------------------------------------------------
# Источник: Semtech SX1276 datasheet, раздел "Error Coding"
# CR=1 → 4/5 (1 бит коррекции на 4 бита данных)
# CR=4 → 4/8 = ½ (4 бита коррекции на 4 бита данных, максимальная защита)

import numpy as np


def encode_cr(bits: np.ndarray, cr: int) -> np.ndarray:
    """
    Упрощённое кодирование с избыточностью согласно Coding Rate.
    Реальный LoRa использует укороченный код Хэмминга (4, 4+cr).

    CR=1: каждые 4 бита → 5 бит (1 бит чётности)
    CR=2: каждые 4 бита → 6 бит (2 бита чётности)
    CR=3: каждые 4 бита → 7 бит (3 бита чётности)
    CR=4: каждые 4 бита → 8 бит (4 бита чётности = повторение)
    """
    out = []
    for i in range(0, len(bits) - len(bits) % 4, 4):
        nibble = bits[i:i+4]
        parity = []
        if cr >= 1: parity.append(nibble[0] ^ nibble[1] ^ nibble[2])
        if cr >= 2: parity.append(nibble[1] ^ nibble[2] ^ nibble[3])
        if cr >= 3: parity.append(nibble[0] ^ nibble[2] ^ nibble[3])
        if cr >= 4: parity.append(nibble[0] ^ nibble[1] ^ nibble[3])
        out.extend(nibble)
        out.extend(parity)
    return np.array(out, dtype=np.uint8)


def decode_cr(bits: np.ndarray, cr: int) -> np.ndarray:
    """Декодирование: берём только информационные биты (первые 4 из кодового слова)."""
    codeword_len = 4 + cr
    out = []
    for i in range(0, len(bits) - len(bits) % codeword_len, codeword_len):
        out.extend(bits[i:i+4])
    return np.array(out, dtype=np.uint8)