# ---------------------------------------------------------------------------
# 1. Генерация чирпа (передатчик)
# ---------------------------------------------------------------------------
# Источник: Semtech patent EP2763321; Chiani & Elzanaty IEEE TWC 2021

from LoRa_Params import LoRaParams
import numpy as np

def generate_chirp(symbol_value: int, p: LoRaParams,
                   up: bool = True) -> np.ndarray:
    """
    Аналитический чирп-символ: N = 2^SF комплексных сэмплов.

    Модель (Semtech patent EP2763321):
        s[n] = exp(j·2π·(c_s·n/BW + sign·BW/(2T)·(n/BW)²))

    где:
        c_s  = (symbol_value/M − 0.5)·BW   — смещение несущей символа
        M    = 2^SF                          — число символов
        T    = M/BW                          — длительность символа [с]
        n    = 0, 1, …, M−1                 — индекс отсчёта
        sign = +1 (up-chirp) или −1 (down-chirp)

    Дискретизация: fs = BW → N = M отсчётов на символ.
    При этом FFT-бин k после де-чирпинга точно равен symbol_value.

    |s[n]| = 1 — константная огибающая (важно для усилителя мощности).
    """
    M    = 2 ** p.sf
    T    = M / p.bw
    N    = M
    t    = np.arange(N) / p.bw                       # время, с
    c_s  = (symbol_value / M - 0.5) * p.bw           # Гц
    sign = 1 if up else -1
    phase = 2 * np.pi * (c_s * t + sign * p.bw / (2 * T) * t ** 2)
    return np.exp(1j * phase)


def generate_base_chirp(p: LoRaParams) -> np.ndarray:
    """Базовый up-chirp (symbol=0) — используется в преамбуле."""
    return generate_chirp(0, p, up=True)


def generate_sync_word(p: LoRaParams) -> np.ndarray:
    """
    Sync word: два down-chirp подряд.
    SX1276: 0x12 — приватная сеть, 0x34 — публичная (LoRaWAN).
    Здесь: упрощённая модель (два нулевых down-chirp).
    """
    d = generate_chirp(0, p, up=False)
    return np.concatenate([d, d])
