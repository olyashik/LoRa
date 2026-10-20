# ---------------------------------------------------------------------------
# plot_per_snr.py — PER(SNR) при разных SF на одном графике, в стиле
# Figure 9 из статьи по LoRa emulator/lab measurement (Semtech-style
# PER curve): линейная шкала PER (не log!), общая ось "чистого" SNR
# (не Eb/N0), пунктирные линии с маркерами по каждому SF.
#
# Только симуляция (без аналитических/lab-кривых поверх).
#
# Замирания (если нужны) включаются централизованно через setup.py
# (cfg.ENABLE_RAYLEIGH / cfg.ENABLE_RICIAN) — этот скрипт их не трогает,
# использует то, что уже настроено в setup.py.
#
# Запуск: python plot_per_snr.py
# ---------------------------------------------------------------------------

import setup as cfg
import math
import numpy as np
import matplotlib.pyplot as plt

from sim_core import compute_ber_per_curve


def _shift_db(sf):
    """Тот же сдвиг SNR -> Eb/N0, что считается внутри sim_core.py —
    нужен здесь только для того, чтобы попросить у compute_ber_per_curve
    единый диапазон SNR (а не Eb/N0), общий для всех SF на графике."""
    rb = sf * (cfg.BW_KHZ * 1e3) / (2 ** sf) * (4 / (4 + cfg.CR))
    return 10 * math.log10((cfg.BW_KHZ * 1e3) / rb)


def plot_per_snr_all_sf(sf_list=(7, 8, 9), n_trials=5000,
                        snr_min=-18, snr_max=-6, snr_step=0.5,
                        save_path="PER_vs_SNR_all_SF.png"):
    """
    Строит PER(SNR) для нескольких SF на одной общей оси SNR (в дБ),
    линейная шкала PER, пунктир+маркеры — визуально в духе Figure 9
    (PER curve for an AWGN channel for SF7,8,9).
    """
    markers = ['o', 's', 'D', '^', 'v', 'P']
    colors  = plt.get_cmap('tab10')

    plt.figure(figsize=(8, 6))

    for i, sf in enumerate(sf_list):
        shift = _shift_db(sf)
        # просим у compute_ber_per_curve такой eb_n0-диапазон, чтобы
        # получившийся snr_range = eb_n0_range - shift совпал с
        # желаемым общим окном [snr_min, snr_max]
        res = compute_ber_per_curve(
            sf, n_trials=n_trials,
            eb_n0_min=snr_min + shift, eb_n0_max=snr_max + shift,
            snr_step=snr_step, verbose=True,
        )

        plt.plot(res["snr_range"], res["per"],
                 linestyle=':', linewidth=1.5,
                 marker=markers[i % len(markers)], markersize=5,
                 markerfacecolor='none', color=colors(i),
                 label=f'SF{sf} (симуляция)')

    plt.xlabel('SNR, дБ', fontsize=12)
    plt.ylabel('PER', fontsize=12)
    plt.title(
        f'PER vs SNR — LoRa, BW={cfg.BW_KHZ} кГц, CR=4/{4+cfg.CR}, '
        f'payload={cfg.PAYLOAD_BYTES} байт\n(только моделирование)',
        fontsize=12
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.ylim([-0.02, 1.02])
    plt.xlim([snr_min, snr_max])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    plot_per_snr_all_sf(sf_list=(7, 8, 9), n_trials=1000,
                        snr_min=-18, snr_max=-6, snr_step=0.5)


if __name__ == "__main__":
    main()