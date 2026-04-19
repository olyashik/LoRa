# ---------------------------------------------------------------------------
# 2. Генерация чирп-сигнала (передатчик)
# ---------------------------------------------------------------------------
# Источник: Chiani & Elzanaty (IEEE TWC 2021), формула (1)-(3)
# Nguyen et al. (IEEE IoT Journal 2019), Section II

import numpy as np
from params import LoRaParams

def generate_chirp(symbol_value: int, p: LoRaParams,
                   fs: float = None, up: bool = True) -> np.ndarray:
    """
    Генерирует один чирп-символ в комплексной форме I + jQ.

    Аналитическая модель (Semtech patent EP2763321):
        s[n] = exp(j·2π·(c_s·n/BW + sign·BW/(2T)·(n/BW)²))

    где:
        c_s  = (symbol_value/M − 0.5)·BW   — смещение несущей символа [Гц]
        M    = 2^SF                          — число символов
        T    = M/BW                          — длительность символа [с]
        n    = 0, 1, …, M−1                 — дискретный индекс отсчёта
        sign = +1 (up-chirp) или −1 (down-chirp)

    Дискретизация: fs = BW → N = M отсчётов на символ.
    При этом после де-чирпинга FFT-бин k точно равен symbol_value.

    Почему не cumsum + % BW:
        Оборачивание частоты через % вносит фазовые разрывы в сигнал.
        De-chirping (FFT) требует непрерывной фазы — иначе пик размывается
        и символ декодируется неверно даже без шума.
        Аналитическая квадратичная фаза разрывов не имеет.

    |s[n]| = 1 — константная огибающая (важно для усилителя мощности).

    Аргументы:
        symbol_value: кодируемый символ (0 .. 2^SF - 1)
        p:  параметры LoRa
        fs: не используется (оставлен для совместимости сигнатуры)
        up: True = up-chirp (данные), False = down-chirp (для демодуляции)
    """
    M    = 2 ** p.sf
    T    = M / p.bw
    N    = M                          # fs = BW → N = M отсчётов на символ
    t    = np.arange(N) / p.bw       # временная ось [с]
    c_s  = (symbol_value / M - 0.5) * p.bw   # смещение несущей [Гц]
    sign = 1 if up else -1
    phase = 2 * np.pi * (c_s * t + sign * p.bw / (2 * T) * t ** 2)
    return np.exp(1j * phase)        # комплексный сигнал единичной амплитуды


def generate_base_chirp(p: LoRaParams, fs: float = None) -> np.ndarray:
    """Базовый up-chirp (symbol=0) — используется для преамбулы."""
    return generate_chirp(0, p, up=True)


def generate_sync_word(p: LoRaParams, fs: float = None) -> np.ndarray:
    """
    Sync word LoRaWAN: символы 0x34 (два даун-чирпа + 2.25 up-чирпа).
    Упрощённая версия: два down-chirp подряд.
    В реальном SX1276 sync word = 0x12 (private) или 0x34 (public/LoRaWAN).
    """
    down = generate_chirp(0, p, up=False)
    return np.concatenate([down, down])