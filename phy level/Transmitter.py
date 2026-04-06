# ---------------------------------------------------------------------------
# 3. Генерация чирп-сигнала (передатчик)
# ---------------------------------------------------------------------------
# Источник: Chiani & Elzanaty (IEEE TWC 2021), формула (1)-(3)
# Nguyen et al. (IEEE IoT Journal 2019), Section II

import LoRa_Params
import numpy as np

def generate_chirp(symbol_value: int, p: LoRa_Params,
                   fs: float = None, up: bool = True) -> np.ndarray:
    '''
    Генерирует один чирп-символ в комплексной форме I + jQ.

    Математическая модель (up-chirp):
        s(t) = exp(j·2π·φ(t))

    Мгновенная частота:
        f(t) = f_start + (B/T)·t,   0 ≤ t < T

    где f_start = (symbol_value / M) * BW — стартовая частота символа,
        M = 2^SF — число возможных символов,
        T = M / BW — длительность символа.

    Фаза (интеграл от f(t)):
        φ(t) = f_start·t + (B / 2T)·t^2

    Сигнал нормализован: |s(t)| = 1 (константная огибающая).
    Это важно для усилителя мощности передатчика (класс C/E).

    Аргументы:
        symbol_value: кодируемый символ (0 .. 2^SF - 1)
        p: параметры LoRa
        fs: частота дискретизации (по умолчанию 8*BW)
        up: True = up-chirp (данные), False = down-chirp (для демодуляции)
    '''
    M  = 2 ** p.sf           # число символов
    T  = M / p.bw            # длительность символа [сек]
    if fs is None:
        fs = 8 * p.bw        # стандартная передискретизация: 8 сэмплов/чирп

    N  = int(round(T * fs))  # число сэмплов на символ
    t  = np.arange(N) / fs   # вектор времени

    # Нормализованная стартовая частота символа (−BW/2 .. +BW/2)
    f_start = (symbol_value / M - 0.5) * p.bw

    # Мгновенная частота с циклическим сдвигом (оборачивается через BW)
    # Реальный чирп обёртывается: когда частота достигает BW/2, она
    # прыгает на −BW/2 и продолжает расти. Это и есть "cyclic shifted chirp".
    sign = 1 if up else -1
    freq_t = (f_start + sign * (p.bw / T) * t) % p.bw - p.bw / 2

    # Фаза: интеграл от freq_t (метод трапеций)
    phase = 2 * np.pi * np.cumsum(freq_t) / fs

    return np.exp(1j * phase)   # комплексный сигнал единичной амплитуды


def generate_base_chirp(p: LoRa_Params, fs: float = None) -> np.ndarray:
    """Базовый up-chirp (symbol=0) — используется для преамбулы."""
    return generate_chirp(0, p, fs, up=True)


def generate_sync_word(p: LoRa_Params, fs: float = None) -> np.ndarray:
    """
    Sync word LoRaWAN: символы 0x34 (два даун-чирпа + 2.25 up-чирпа).
    Упрощённая версия: два down-chirp подряд.
    В реальном SX1276 sync word = 0x12 (private) или 0x34 (public/LoRaWAN).
    """
    down = generate_chirp(0, p, fs, up=False)
    return np.concatenate([down, down])
