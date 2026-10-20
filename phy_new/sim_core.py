# ---------------------------------------------------------------------------
# sim_core.py — общая Monte-Carlo симуляция канала LoRa.
#
# Считает BER(Eb/N0) и PER(Eb/N0) ОДНОВРЕМЕННО за один проход, чтобы
# plot_ber.py и plot_per.py не дублировали дорогой цикл симуляции —
# каждый скрипт просто берёт из результата нужное поле ("ber" или "per").
#
# PER — доля пакетов, в которых есть хотя бы один битовый эрор
# (стандартное определение, см. Semtech AN1200.22 "LoRa Modulation Basics").
# ---------------------------------------------------------------------------

import setup as cfg
import math
import numpy as np

from params import LoRaParams
from phy import phy_transmit, phy_receive
from channel import simulate_channel
from BER_teor import snr_threshold_db


def compute_ber_per_curve(sf, n_trials=10000, eb_n0_min=-15, eb_n0_max=10,
                          snr_step=1, verbose=True):
    """
    Считает BER(Eb/N0) и PER(Eb/N0) для заданного Spreading Factor.

    Замирания (Rayleigh/Rician) включаются/выключаются и настраиваются
    централизованно через setup.py (cfg.ENABLE_RAYLEIGH, cfg.ENABLE_RICIAN,
    cfg.K_FACTOR_DB, cfg.DOPPLER_HZ) — как и остальные ENABLE_*-флаги
    канала, без отдельных аргументов функции.

    Возвращает словарь:
        sf, rb, shift_db, snr_range, eb_n0_range, ber, per, threshold_db
    """
    p = LoRaParams(
        sf               = sf,
        bw               = cfg.BW_KHZ * 1e3,
        cr               = cfg.CR,
        tx_power_dbm     = cfg.TX_POWER_DBM,
        freq_hz          = cfg.FREQ_MHZ * 1e6,
        preamble_symbols = cfg.PREAMBLE_SYMS,
        explicit_header  = cfg.EXPLICIT_HDR,
    )

    if not cfg.ENABLE_LDR_AUTO:
        p.low_dr_opt = False

    # ── Битовая скорость и сдвиг SNR → Eb/N0 ─────────────────────────────
    rb       = sf * (cfg.BW_KHZ * 1e3) / (2 ** sf) * (4 / (4 + cfg.CR))
    shift_db = 10 * math.log10((cfg.BW_KHZ * 1e3) / rb)

    if verbose:
        print(f"\n=== SF={sf}, BW={cfg.BW_KHZ} кГц, CR=4/{4+cfg.CR} ===")
        print(f"Rb = {rb:.1f} бит/с")
        print(f"Сдвиг SNR → Eb/N0 = {shift_db:.1f} дБ")

    # ── Диапазон SNR ──────────────────────────────────────────────────────
    snr_range = np.arange(eb_n0_min - shift_db, eb_n0_max - shift_db, snr_step)

    if verbose:
        print(f"SNR range:   {snr_range[0]:.1f} .. {snr_range[-1]:.1f} дБ")
        print(f"Eb/N0 range: {snr_range[0]+shift_db:.1f} .. {snr_range[-1]+shift_db:.1f} дБ")

    # ── Основной цикл BER/PER ────────────────────────────────────────────
    BER_all, PER_all = [], []
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    for SNR in snr_range:
        BER, PER = [], []

        for N in range(n_trials):
            n_bits   = cfg.PAYLOAD_BYTES * 8
            tx_bits  = rng.integers(0, 2, n_bits, dtype=np.uint8)
            tx_bytes = np.packbits(tx_bits).tobytes()

            tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=cfg.ENABLE_TOA)

            rx_signal, _ = simulate_channel(
                tx_signal, p,
                distance_m      =cfg.DISTANCE_M,
                noise_figure_db =cfg.NOISE_FIGURE_DB,
                temperature_k   =cfg.TEMPERATURE_K,
                path_loss_exp   =cfg.PATH_LOSS_EXP,
                enable_awgn     =cfg.ENABLE_AWGN,
                enable_path_loss=cfg.ENABLE_PATH_LOSS,
                fixed_snr_db    =SNR,
                enable_rayleigh =cfg.ENABLE_RAYLEIGH,
                enable_rician   =cfg.ENABLE_RICIAN,
                k_factor_db     =cfg.K_FACTOR_DB,
                doppler_hz      =cfg.DOPPLER_HZ,
                fading_seed     =None,   # новая реализация замирания на каждый пакет
            )

            rx_data, _ = phy_receive(
                rx_signal, p,
                len(tx_bytes),
                enable_ToA=cfg.ENABLE_TOA,
            )

            rx_bytes_padded = rx_data.ljust(len(tx_bytes), b'\x00')
            rx_bits = np.unpackbits(
                np.frombuffer(rx_bytes_padded, dtype=np.uint8)
            )[:n_bits]

            n_errors = int(np.sum(tx_bits ^ rx_bits))
            BER.append(n_errors / n_bits)
            PER.append(1.0 if n_errors > 0 else 0.0)

        BER_all.append(np.mean(BER))
        PER_all.append(np.mean(PER))

        if verbose:
            print(f"SF={sf} | SNR={SNR:+.1f} дБ | Eb/N0={SNR+shift_db:+.1f} дБ | "
                  f"BER={BER_all[-1]:.4e} | PER={PER_all[-1]:.3f}")

    BER_all = np.array(BER_all)
    PER_all = np.array(PER_all)
    eb_n0_range = snr_range + shift_db
    threshold_db = snr_threshold_db(p) + shift_db

    return {
        "sf": sf, "rb": rb, "shift_db": shift_db,
        "snr_range": snr_range, "eb_n0_range": eb_n0_range,
        "ber": BER_all, "per": PER_all,
        "threshold_db": threshold_db,
    }


def compute_all_sf(sf_list=range(7, 13), n_trials=1000,
                   eb_n0_min=-15, eb_n0_max=10, snr_step=1, verbose=True):
    """Прогоняет compute_ber_per_curve для списка SF, возвращает список результатов."""
    return [
        compute_ber_per_curve(sf, n_trials=n_trials, eb_n0_min=eb_n0_min,
                              eb_n0_max=eb_n0_max, snr_step=snr_step,
                              verbose=verbose)
        for sf in sf_list
    ]