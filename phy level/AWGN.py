# ---------------------------------------------------------------------------
# 6. AWGN канал + затухание
# ---------------------------------------------------------------------------
# Источник: формула Фрииса для свободного пространства +
# упрощённая логарифмическая модель потерь

import numpy as np
from typing import Tuple
import math
from LoRa_Params import LoRaParams

def path_loss_db(distance_m: float, freq_hz: float,
                 path_loss_exp: float = 2.7) -> float:
    """
    Модель логарифмических потерь распространения.

    Формула Фрииса (свободное пространство, n=2):
        PL_0 = 20·log10(4π·d·f/c)   [дБ]

    Общая модель (городская среда, n=2.7..4.0):
        PL = PL_0 + 10·n·log10(d/d_0)

    где d_0 = 1 м, PL_0 = потери на 1 м (из формулы Фрииса)
    """
    c = 3e8
    pl_0 = 20 * math.log10(4 * math.pi * freq_hz / c)  # потери на 1 м в свободном пространстве
    return pl_0 + 10 * path_loss_exp * math.log10(max(distance_m, 1)) # потери на 1 м в городской среде


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Добавляет комплексный белый гауссовский шум (AWGN).

    SNR = P_signal / P_noise = 10^(snr_db/10)
    σ² = P_signal / SNR        (дисперсия шума)
    σ  = sqrt(P_signal / SNR)  (среднеквадратическое отклонение)

    Шум комплексный: n = (n_I + j·n_Q) / √2, оба компонента N(0,σ²/2)
    """
    p_signal = np.mean(np.abs(signal) ** 2)
    snr_lin  = 10 ** (snr_db / 10)
    sigma    = math.sqrt(p_signal / snr_lin)
    noise    = (np.random.randn(len(signal)) +
                1j * np.random.randn(len(signal))) * sigma / math.sqrt(2)
    return signal + noise


def simulate_channel(signal: np.ndarray, p: LoRaParams,
                     distance_m: float      = 2000.0,
                     noise_figure_db: float = 6.0,
                     temperature_k: float   = 290.0,
                     path_loss_exp: float   = 2.7,
                     enable_awgn: bool      = True,
                     enable_path_loss: bool = True,
                     fixed_snr_db: float    = 10.0) -> Tuple[np.ndarray, float]:
    """
    Модель канала с управляемыми флагами из lora_setup.py.

    enable_path_loss = False → SNR = fixed_snr_db
    enable_awgn      = False → сигнал без шума (SNR = +∞)

    Тепловой шум (Джонсон–Найквист):
        N₀ = k·T·B  [Вт],  N₀_dBm = 10·log10(k·T·B) + 30
    """
    if not enable_path_loss:
        snr_db = fixed_snr_db
    else:
        k_b    = 1.38e-23
        n0_dbm = 10 * math.log10(k_b * temperature_k * p.bw) + 30
        pl     = path_loss_db(distance_m, p.freq_hz, path_loss_exp)
        p_rx   = p.tx_power_dbm - pl
        snr_db = p_rx - n0_dbm - noise_figure_db

    if not enable_awgn:
        return signal.copy(), snr_db

    return add_awgn(signal, snr_db), snr_db
