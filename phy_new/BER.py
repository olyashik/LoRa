import setup as cfg
import math
import numpy as np
import matplotlib.pyplot as plt

from LoRa_ToA import *
from params import LoRaParams
from modulate import *
from demodulate import *
from LoRa_Chirp import *
from LoRa_Coding import *
from BER_teor import *
from channel import *
from phy import *


def compute_ber_curve(sf, n_trials=1000, eb_n0_min=-15, eb_n0_max=10, snr_step=1):
    """
    Считает BER(Eb/N0) для заданного Spreading Factor.

    Возвращает словарь с результатами: sf, rb, shift_db,
    snr_range, eb_n0_range, ber, threshold_db.
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

    print(f"\n=== SF={sf}, BW={cfg.BW_KHZ} кГц, CR=4/{4+cfg.CR} ===")
    print(f"Rb = {rb:.1f} бит/с")
    print(f"Сдвиг SNR → Eb/N0 = {shift_db:.1f} дБ")

    # ── Диапазон SNR ──────────────────────────────────────────────────────
    snr_range = np.arange(eb_n0_min - shift_db, eb_n0_max - shift_db, snr_step)

    print(f"SNR range:   {snr_range[0]:.1f} .. {snr_range[-1]:.1f} дБ")
    print(f"Eb/N0 range: {snr_range[0]+shift_db:.1f} .. {snr_range[-1]+shift_db:.1f} дБ")

    # ── Основной цикл BER ────────────────────────────────────────────────
    BER_all = []
    rng     = np.random.default_rng(cfg.RANDOM_SEED)

    for SNR in snr_range:
        BER = []

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

        BER_all.append(np.mean(BER))
        print(f"SF={sf} | SNR={SNR:+.1f} дБ | Eb/N0={SNR+shift_db:+.1f} дБ | BER={BER_all[-1]:.4e}")

    BER_all = np.array(BER_all)
    eb_n0_range = snr_range + shift_db
    threshold_db = snr_threshold_db(p) + shift_db

    return {
        "sf": sf,
        "rb": rb,
        "shift_db": shift_db,
        "snr_range": snr_range,
        "eb_n0_range": eb_n0_range,
        "ber": BER_all,
        "threshold_db": threshold_db,
    }


def plot_all_sf(results, save_path="BER_result_all_SF.png"):
    """Рисует BER(Eb/N0) для всех переданных SF на одном графике."""
    plt.figure(figsize=(10, 7))

    cmap = plt.get_cmap('tab10')

    for i, res in enumerate(results):
        color = cmap(i % 10)
        plt.semilogy(
            res["eb_n0_range"], res["ber"],
            marker='o', linewidth=2, markersize=1,
            color=color, label=f'SF{res["sf"]} (симуляция)'
        )

    eb_n0_all = np.concatenate([r["eb_n0_range"] for r in results])

    plt.xlabel('Eb/N0 (дБ)', fontsize=12)
    plt.ylabel('BER', fontsize=12)
    plt.title(
        f'BER vs Eb/N0 — LoRa SF7..SF12, BW{cfg.BW_KHZ}kHz, CR4/{4+cfg.CR}',
        fontsize=12
    )
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, which='both', alpha=0.4)
    plt.ylim([1e-4, 1])
    plt.xlim([eb_n0_all.min(), eb_n0_all.max()])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main(snr_step=0.1):
    sf_list = range(7, 13)  # SF7..SF12
    results = []

    for sf in sf_list:
        res = compute_ber_curve(sf, snr_step=snr_step)
        results.append(res)

    plot_all_sf(results)


if __name__ == "__main__":
    main(snr_step=0.1)