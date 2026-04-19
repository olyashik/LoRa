# ---------------------------------------------------------------------------
# 8. Анализ битовых ошибок
# ---------------------------------------------------------------------------

import numpy as np


def bits_from_bytes(data: bytes) -> np.ndarray:
    """bytes → массив бит (MSB первый)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def analyze_errors(tx_bits: np.ndarray, rx_bits: np.ndarray,
                   map_width: int = 64) -> dict:
    """
    Полный анализ ошибок одного пакета.

    Возвращает:
        n_bits          — всего бит
        n_errors        — ошибочных бит
        ber             — BER = n_errors / n_bits
        packet_ok       — True если ошибок нет
        error_positions — индексы ошибочных бит
        error_mask      — XOR tx ^ rx
        bit_map         — карта '.' / 'X' по map_width бит в строке
    """
    n    = min(len(tx_bits), len(rx_bits))
    mask = tx_bits[:n] ^ rx_bits[:n]
    pos  = [int(i) for i in np.where(mask == 1)[0]]

    lines = []
    chars = ["X" if b else "." for b in mask]
    for i in range(0, len(chars), map_width):
        row = chars[i: i + map_width]
        lines.append("".join(row) + f"  [{row.count('X'):3d} ош.]")

    return {
        "n_bits":          n,
        "n_errors":        int(np.sum(mask)),
        "ber":             float(np.sum(mask)) / n if n > 0 else 0.0,
        "packet_ok":       int(np.sum(mask)) == 0,
        "error_positions": pos,
        "error_mask":      mask,
        "bit_map":         "\n".join(lines),
    }