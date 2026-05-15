import numpy as np
from typing import Tuple
import math
from params import LoRaParams
from LoRa_Chirp import *
from LoRa_ToA import *
from LoRa_Coding import *
from gray import *


def demodulate_symbol(rx_chirp: np.ndarray, p: LoRaParams,
                      fs: float = None) -> int:
    """
    Декодирует один символ из принятого чирпа.
    Де-чирпинг: умножаем на conj(base_up_chirp), берём argmax FFT.
    """
    M    = 2 ** p.sf
    base = generate_chirp(0, p, up=True)

    if len(rx_chirp) < M:
        return 0

    dechirped = rx_chirp[:M] * np.conj(base)
    spectrum  = np.abs(np.fft.fft(dechirped, n=M)) ** 2
    return int(np.argmax(spectrum)) % M


def demodulate(symbols: np.ndarray, p: LoRaParams) -> bytes:
    """
    Символы → байты. Строго зеркально modulate().

    modulate():
        биты → encode_cr → pad → reshape(-1, sf) → .T.flatten()
             → gray_encode → символы

    demodulate():
        символы → gray_decode → биты
               → reshape(sf, -1) → .T.flatten()   # зеркало .T.flatten() после reshape(-1,sf)
               → decode_cr → байты
    """
    sf    = p.sf
    M     = 2 ** sf
    bits  = []

    # Символы → биты (с Gray decode) — зеркало последнего шага modulate
    for sym in symbols:
        sym      = int(sym) % M
        sym      = gray_decode(sym)
        sym_bits = [(sym >> (sf - 1 - i)) & 1 for i in range(sf)]
        bits.extend(sym_bits)

    bits  = np.array(bits, dtype=np.uint8)
    n_sym = len(symbols)

    if n_sym == 0:
        return b''

    # Де-перемежение — строго зеркально modulate:
    #   modulate:   reshape(-1, sf)  → .T → flatten  →  форма (sf, n_sym) развёрнутая
    #   demodulate: reshape(sf, n_sym) → .T → flatten
    #
    # Проверка: если modulate даёт X = M.T.flatten() где M.shape=(n,sf),
    #           то demodulate: X.reshape(sf, n).T.flatten() восстанавливает исходное
    matrix        = bits.reshape(sf, n_sym)   # (sf, n_sym)
    deinterleaved = matrix.T.flatten()        # (n_sym * sf,) — исходный порядок

    # FEC декодирование — зеркало encode_cr
    decoded = decode_cr(deinterleaved, p.cr)

    # Биты → байты
    pad = (-len(decoded)) % 8
    if pad:
        decoded = np.append(decoded, np.zeros(pad, dtype=np.uint8))

    return np.packbits(decoded).tobytes()
