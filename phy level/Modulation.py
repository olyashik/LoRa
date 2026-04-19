# ---------------------------------------------------------------------------
# 4. Модуляция: байты → символы → чирпы (передатчик)
# ---------------------------------------------------------------------------

from LoRa_Params import LoRaParams
from Chirp import *
from FEC import *


def modulate(symbols: np.ndarray, p: LoRaParams) -> np.ndarray:
    """
    Символы → I/Q сигнал.

    Физический фрейм:
        [преамбула: N_pre up-chirp] [sync word: 2 down-chirp] [данные]

    N = M = 2^SF отсчётов на символ во всём сигнале.
    """
    base  = generate_base_chirp(p)
    parts = [base.copy() for _ in range(p.preamble_symbols)]
    parts.append(generate_sync_word(p))
    for sym in symbols:
        parts.append(generate_chirp(int(sym), p, up=True))
    return np.concatenate(parts)