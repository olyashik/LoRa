# ---------------------------------------------------------------------------
# Модуляция: байты → символы → чирпы 
# ---------------------------------------------------------------------------

import numpy as np
from params import LoRaParams
from LoRa_Chirp import *
from LoRa_Coding import *
from gray import *


def modulate(data: bytes, p: LoRaParams) -> np.ndarray:
    """
    Преобразует байты в LoRa-символы.

    Шаги:
    1. Байты → биты
    2. Кодирование FEC (Coding Rate)
    3. Перемежение (interleaving) — защита от пакетных ошибок
    4. Биты → символы (SF бит на символ)

    Число символов = ceil(len(encoded_bits) / SF)
    """
    # Байты → биты (MSB первый)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # FEC кодирование
    encoded = encode_cr(bits, p.cr)

    # Простое перемежение: читаем биты по диагонали (SF×SF матрица)
    sf  = p.sf
    pad = (-len(encoded)) % sf
    if pad:
        encoded = np.append(encoded, np.zeros(pad, dtype=np.uint8))
    matrix      = encoded.reshape(-1, sf)
    interleaved = matrix.T.flatten()

    # Биты → символы (каждые SF бит = один символ)
    symbols = []
    for i in range(0, len(interleaved), sf):
        chunk = interleaved[i:i+sf]
        if len(chunk) == sf:
            sym = int(''.join(map(str, chunk)), 2) % (2 ** sf)
            # Gray encoding
            sym = gray_encode(sym)
            symbols.append(sym)

    return np.array(symbols, dtype=np.int32)


