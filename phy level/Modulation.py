# ---------------------------------------------------------------------------
# 5. Модуляция: байты → символы → чирпы (передатчик)
# ---------------------------------------------------------------------------

import LoRa_Params
from Transmitter import *
from FEC import *

def bytes_to_symbols(data: bytes, p: LoRa_Params) -> np.ndarray:
    '''
    Преобразует байты в LoRa-символы.

    Шаги:
    1. Байты → биты
    2. Кодирование FEC (Coding Rate)
    3. Перемежение (interleaving) — защита от пакетных ошибок
    4. Биты → символы (SF бит на символ)

    Число символов = ceil(len(encoded_bits) / SF)
    '''
    # Байты → биты
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # FEC кодирование
    encoded = encode_cr(bits, p.cr)

    # Простое перемежение: читаем биты по диагонали (SF×SF матрица)
    # Реальное перемежение в LoRa сложнее, но принцип тот же
    sf = p.sf
    pad = (-len(encoded)) % sf
    if pad:
        encoded = np.append(encoded, np.zeros(pad, dtype=np.uint8))
    matrix = encoded.reshape(-1, sf)
    interleaved = matrix.T.flatten()

    # Биты → символы (каждые SF бит = один символ)
    symbols = []
    for i in range(0, len(interleaved), sf):
        chunk = interleaved[i:i+sf]
        if len(chunk) == sf:
            b = int(''.join(map(str, chunk)), 2) % (2 ** sf)
            sym = binary_to_gray(b)
            symbols.append(sym)

    return np.array(symbols, dtype=np.int32)


def modulate(data: bytes, p: LoRa_Params, fs: float = None) -> np.ndarray:
    '''
    Полная модуляция: байты → радиосигнал I+jQ.

    Структура пакета (физический фрейм):
        [преамбула] [sync word] [заголовок] [полезная нагрузка]

    Преамбула: N up-chirp'ов (N = preamble_symbols, стандарт = 8)
    Sync word: 2 down-chirp'а (идентификатор сети LoRaWAN)
    Далее: символы данных
    '''
    if fs is None:
        fs = 8 * p.bw

    base  = generate_base_chirp(p, fs)
    parts = []

    # Преамбула: preamble_symbols up-chirp'ов
    for _ in range(p.preamble_symbols):
        parts.append(base.copy())

    # Sync word (2 down-chirp)
    parts.append(generate_sync_word(p, fs))

    # Символы данных
    symbols = bytes_to_symbols(data, p)
    for sym in symbols:
        parts.append(generate_chirp(sym, p, fs))

    return np.concatenate(parts)