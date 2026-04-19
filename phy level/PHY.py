"""
lora_phy.py — физический уровень LoRa на Python
================================================
Реализует модель передатчика и приёмника по формулам:
  - Semtech AN1200.22 (LoRa Modulation Basics)
  - Semtech SX1276 datasheet
  - Semtech patent EP2763321 (de-chirping algorithm)
  - Chiani & Elzanaty, IEEE TWC 2021 (математика CSS)
  - Elshabrawy & Robert, IEEE Comm. Letters 2018 (BER formula)

Зависимости:
  lora_params.py — класс LoRaParams
  lora_setup.py  — константы симуляции

Ключевые соглашения:
  N = 2^SF сэмплов на символ (fs = BW, oversampling = 1).
  Это единственное значение N во всей цепочке.
"""

import numpy as np
import math
from typing import Tuple, Optional
from scipy.special import erfc

from LoRa_Params import LoRaParams
from BER import *
import lora_setup as cfg
from AWGN import *
from Coding_sim_bits import *
from Modulation import *
from Demodulation import *
from Chirp import *

if __name__ == "__main__":

    # --- Параметры ---
    p = LoRaParams(
        sf=cfg.SF,
        bw=cfg.BW_KHZ * 1e3,
        cr=cfg.CR,
        tx_power_dbm=cfg.TX_POWER_DBM,
        freq_hz=cfg.FREQ_MHZ * 1e6,
        preamble_symbols=cfg.PREAMBLE_SYMS,
        explicit_header=cfg.EXPLICIT_HDR,
    )

    rng = np.random.default_rng(cfg.RANDOM_SEED)

    print("=" * 60)
    print("LoRa PHY — демонстрация передачи")
    print("=" * 60)

    # --- Конфигурация ---
    print("\n[ПАРАМЕТРЫ]")
    print(f"SF={p.sf}, BW={p.bw/1e3:.0f} кГц, CR=4/{4+p.cr}")
    print(f"Частота: {p.freq_hz/1e6:.1f} МГц")
    print(f"Длина сообщения: {cfg.MSG_LEN_BITS} бит")

    # --- Генерация ---
    bits = rng.integers(0, 2, cfg.MSG_LEN_BITS, dtype=np.uint8)

    n_sym = len(bits) // p.sf
    bits = bits[:n_sym * p.sf]

    tx_syms = bits_to_symbols(bits, p.sf, gray=True)

    print("\n[ПЕРЕДАЧА]")
    print(f"Биты (первые 32): {''.join(map(str, bits[:32]))}...")
    print(f"Символы (первые 8): {tx_syms[:8]}")

    # --- Канал ---
    tx_signal = modulate(tx_syms, p)
    rx_signal, snr = simulate_channel(
        tx_signal, p,
        distance_m=cfg.DISTANCE_M,
        enable_awgn=cfg.ENABLE_AWGN,
        enable_path_loss=cfg.ENABLE_PATH_LOSS,
        fixed_snr_db=cfg.FIXED_SNR_DB
    )

    print("\n[КАНАЛ]")
    print(f"SNR = {snr:.2f} дБ")

    # --- Приём ---
    rx_syms = demodulate(rx_signal, p, len(tx_syms))
    rx_bits = symbols_to_bits(rx_syms, p.sf, gray=True)

    print("\n[ПРИЁМ]")
    print(f"Биты (первые 32): {''.join(map(str, rx_bits[:32]))}...")
    print(f"Символы (первые 8): {rx_syms[:8]}")

    # --- Ошибки ---
    bit_errors = np.sum(rx_bits != bits)
    sym_errors = np.sum(rx_syms != tx_syms)

    print("\n[ОШИБКИ]")
    print(f"Ошибки бит: {bit_errors} / {len(bits)}")
    print(f"Ошибки символов: {sym_errors} / {len(tx_syms)}")

    # --- BER кривая ---
    print("\n[BER СИМУЛЯЦИЯ]")
    EbN0dB = np.arange(0, 12, 2)

    ber = simulate_ber(
        p,
        msg_len=cfg.MSG_LEN_BITS,
        EbN0dB=EbN0dB,
        NumExp=500
    )

    for snr_db, b in zip(EbN0dB, ber):
        print(f"SNR={snr_db:2d} дБ → BER={b:.4e}")

    print("\nГотово.")