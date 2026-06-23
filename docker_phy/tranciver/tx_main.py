"""
tx_main.py — передатчик (Transmitter)
Генерирует один пакет, кодирует, модулирует, накладывает шум канала,
отправляет по UDP на приёмник.
"""

import setup as cfg
import numpy as np

from params import LoRaParams
from phy import phy_transmit
from channel import simulate_channel
from transport import UDPFrameSender, pack_frame

RX_HOST = "192.168.100.2"      # имя сервиса в docker-compose (для GNS3 заменить на статический IP, напр. "192.168.100.2")
RX_PORT = 5005

p = LoRaParams(
    sf=cfg.SF, bw=cfg.BW_KHZ * 1e3, cr=cfg.CR,
    tx_power_dbm=cfg.TX_POWER_DBM, freq_hz=cfg.FREQ_MHZ * 1e6,
    preamble_symbols=cfg.PREAMBLE_SYMS, explicit_header=cfg.EXPLICIT_HDR,
)
if not cfg.ENABLE_LDR_AUTO:
    p.low_dr_opt = False

rng = np.random.default_rng(cfg.RANDOM_SEED)

n_bits = cfg.PAYLOAD_BYTES * 8
tx_bits = rng.integers(0, 2, n_bits, dtype=np.uint8)
tx_bytes = np.packbits(tx_bits).tobytes()

print(f"[TX] Сгенерирован пакет: {cfg.PAYLOAD_BYTES} байт ({n_bits} бит)")

tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=cfg.ENABLE_TOA)
print(f"[TX] Сигнал смодулирован: {len(tx_signal)} отсчётов")

rx_signal, _ = simulate_channel(
    tx_signal, p,
    distance_m=cfg.DISTANCE_M,
    noise_figure_db=cfg.NOISE_FIGURE_DB,
    temperature_k=cfg.TEMPERATURE_K,
    path_loss_exp=cfg.PATH_LOSS_EXP,
    enable_awgn=cfg.ENABLE_AWGN,
    enable_path_loss=cfg.ENABLE_PATH_LOSS,
    fixed_snr_db=cfg.FIXED_SNR_DB,
)
print(f"[TX] Канал применён, SNR={cfg.FIXED_SNR_DB} дБ")

payload = pack_frame(
    snr_db=float(cfg.FIXED_SNR_DB),
    n_bits=n_bits,
    tx_bits=tx_bits,
    rx_signal=rx_signal,
    snr_idx=0, total_snrs=1,
    iter_idx=0, total_iters=1,
)

sender = UDPFrameSender(RX_HOST, RX_PORT)
sender.send(frame_id=0, payload=payload)
sender.close()

print("[TX] Пакет отправлен")