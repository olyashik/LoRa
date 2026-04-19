# ---------------------------------------------------------------------------
# 2. Демодуляция одного символа (приёмник)
# ---------------------------------------------------------------------------
# Источник: Semtech patent EP2763321

from LoRa_Params import LoRaParams
from Chirp import *

def demodulate_symbol(rx: np.ndarray, p: LoRaParams) -> int:
    """
    De-chirping + M-точечный FFT → символ.

    Алгоритм:
        1. base[n] = exp(j·2π·(c_0·n/BW + BW/(2T)·(n/BW)²))  — up-chirp sym=0
        2. y[n]    = rx[n] · conj(base[n])                      — дечирпинг
        3. Y[k]    = FFT(y, n=M)
        4. symbol  = argmax |Y[k]|²

    После де-чирпинга y[n] = exp(j·2π·(c_s − c_0)·n/BW)
                            = exp(j·2π·(sym/M)·n)
    — чистый тон, FFT-пик точно на бине k = symbol_value.

    Processing Gain: PG = 10·log10(M) дБ
        SF=7:  PG ≈ 21 дБ
        SF=12: PG ≈ 36 дБ
    Именно поэтому LoRa работает при SNR ниже нуля.
    """
    M    = 2 ** p.sf
    N    = M
    base = generate_chirp(0, p, up=True)
    y    = rx[:N] * np.conj(base)
    spec = np.abs(np.fft.fft(y, n=M)) ** 2
    return int(np.argmax(spec)) % M

# ---------------------------------------------------------------------------
# Демодуляция пакета
# ---------------------------------------------------------------------------

def demodulate(rx_signal: np.ndarray, p: LoRaParams,
               n_symbols: int) -> np.ndarray:
    """
    I/Q сигнал → массив символов.

    Пропускает преамбулу и sync word, декодирует n_symbols символов.
    N = M = 2^SF отсчётов на символ.
    """
    M      = 2 ** p.sf
    N      = M
    skip   = (p.preamble_symbols + 2) * N    # преамбула + 2 sync
    rx     = rx_signal[skip:]
    result = np.zeros(n_symbols, dtype=np.int32)
    for i in range(min(n_symbols, len(rx) // N)):
        result[i] = demodulate_symbol(rx[i * N: (i + 1) * N], p)
    return result
