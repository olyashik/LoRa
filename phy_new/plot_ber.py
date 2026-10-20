# ---------------------------------------------------------------------------
# plot_ber.py — строит BER(Eb/N0) для SF7..SF12.
# Запуск: python plot_ber.py
# ---------------------------------------------------------------------------

import setup as cfg
import numpy as np
import matplotlib.pyplot as plt

from sim_core import compute_all_sf


def plot_all_sf(results, save_path="BER_result_all_SF.png"):
    """Рисует BER(Eb/N0) для всех переданных SF на одном графике."""
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap('tab10')

    for i, res in enumerate(results):
        color = cmap(i % 10)
        plt.semilogy(
            res["eb_n0_range"], res["ber"],
            marker='o', linewidth=2, markersize=3,
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
    plt.ylim([1e-5, 1])
    plt.xlim([eb_n0_all.min(), eb_n0_all.max()])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main(snr_step=0.5):
    sf_list = range(7, 13)
    results = compute_all_sf(sf_list=sf_list, snr_step=snr_step)
    plot_all_sf(results)


if __name__ == "__main__":
    main(snr_step=0.5)