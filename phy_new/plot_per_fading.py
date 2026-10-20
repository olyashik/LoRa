# ---------------------------------------------------------------------------
# plot_per_fading.py — PER(SNR) для одного SF: AWGN vs AWGN+Rayleigh.
#
# По структуре повторяет Figure 10 из Elshabrawy & Robert,
# "On the Error Rate of the LoRa Modulation with Interference",
# arXiv:1905.11252 (Rayleigh-канал даёт пологий "хвост" на десятки дБ
# из-за случайных глубоких замираний отдельных пакетов, в отличие от
# резкого порога чистого AWGN) — но здесь ТОЛЬКО симуляция, без
# аналитических кривых поверх.
#
# Запуск: python plot_per_fading.py
# ---------------------------------------------------------------------------

import setup as cfg
import numpy as np
import matplotlib.pyplot as plt

from sim_core import compute_ber_per_curve


def plot_awgn_vs_rayleigh(sf=7, n_trials=5000, snr_min=-15, snr_max=20,
                          snr_step=1.0, k_factor_db=6.0,
                          save_path="PER_AWGN_vs_Rayleigh.png"):
    """
    Считает и рисует PER(SNR) для трёх каналов на одном SF:
      - AWGN (без замираний)
      - AWGN + Rayleigh fading (NLOS)
      - AWGN + Rician fading (K=k_factor_db дБ) — бонус, для сравнения
        промежуточного случая между AWGN и чистым Rayleigh.

    Замирания включаются/выключаются централизованно через setup.py
    (cfg.ENABLE_RAYLEIGH / cfg.ENABLE_RICIAN / cfg.K_FACTOR_DB), поэтому
    здесь скрипт временно переключает эти флаги перед каждым прогоном
    и восстанавливает исходные значения в конце.

    Использует SNR "как есть" (eb_n0_min/max трактуются как SNR, без
    учёта сдвига Eb/N0), чтобы напрямую сравниться по оси с Figure 10
    оригинальной статьи.
    """
    common = dict(sf=sf, n_trials=n_trials,
                 eb_n0_min=snr_min, eb_n0_max=snr_max, snr_step=snr_step,
                 verbose=True)

    # сохраняем исходное состояние флагов setup.py, чтобы вернуть их после
    orig = (cfg.ENABLE_RAYLEIGH, cfg.ENABLE_RICIAN, cfg.K_FACTOR_DB)

    try:
        print("=== AWGN ===")
        cfg.ENABLE_RAYLEIGH, cfg.ENABLE_RICIAN = False, False
        res_awgn = compute_ber_per_curve(**common)

        print("\n=== AWGN + Rayleigh ===")
        cfg.ENABLE_RAYLEIGH, cfg.ENABLE_RICIAN = True, False
        res_rayleigh = compute_ber_per_curve(**common)

        print(f"\n=== AWGN + Rician (K={k_factor_db} дБ) ===")
        cfg.ENABLE_RAYLEIGH, cfg.ENABLE_RICIAN = False, True
        cfg.K_FACTOR_DB = k_factor_db
        res_rician = compute_ber_per_curve(**common)
    finally:
        cfg.ENABLE_RAYLEIGH, cfg.ENABLE_RICIAN, cfg.K_FACTOR_DB = orig

    # shift_db одинаков для всех трёх (один и тот же SF/BW/CR), поэтому
    # snr_range у них совпадает — используем ось "чистого" SNR, а не Eb/N0,
    # чтобы соответствовать оси Figure 10.
    snr_range = res_awgn["snr_range"]

    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_range, res_awgn["per"], 'k--', marker='s', markersize=4,
                linewidth=1.5, label='AWGN (симуляция)')
    plt.semilogy(snr_range, res_rayleigh["per"], 'b-', marker='D', markersize=4,
                linewidth=1.5, label='AWGN + Rayleigh (симуляция)')
    plt.semilogy(snr_range, res_rician["per"], 'g-', marker='^', markersize=4,
                linewidth=1.5, label=f'AWGN + Rician, K={k_factor_db} дБ (симуляция)')

    plt.xlabel('SNR, дБ', fontsize=12)
    plt.ylabel('PER', fontsize=12)
    plt.title(f'PER vs SNR — LoRa SF{sf}, BW{cfg.BW_KHZ} кГц, CR4/{4+cfg.CR}\n'
              f'AWGN vs Rayleigh vs Rician (только моделирование)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', alpha=0.4)
    plt.ylim([1e-3, 1.2])
    plt.xlim([snr_min, snr_max])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    return res_awgn, res_rayleigh, res_rician


def main():
    plot_awgn_vs_rayleigh(sf=7, n_trials=5000, snr_min=-15, snr_max=20, snr_step=1.0)


if __name__ == "__main__":
    main()