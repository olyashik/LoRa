# ---------------------------------------------------------------------------
# plot_per_chain_sir.py — PER от уровня гармонической помехи (SIR)
# по цепочке TX → канал → RX для SF7..SF12.
#
# Помеха — тон на смещении setup.HARMONIC_FREQ_OFFSET_HZ внутри полосы
# (число гармоник и затухание — HARMONIC_N, HARMONIC_DECAY). SNR фиксирован
# выше порога SF (по умолчанию +3 дБ), поэтому без помехи PER ≈ 0, а потери
# вызваны только тоном. Крестик — SIR, при котором PER = 10 %.
#
# Запуск из phy_new:
#   python plot_per_chain_sir.py
#   python plot_per_chain_sir.py --base rayleigh --snr-offset 10   # тон + Рэлей
#   python plot_per_chain_sir.py --sir-lo -35 --sir-hi 0 --sir-step 2.5
# ---------------------------------------------------------------------------

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import setup as cfg
from extra_common import base_parser, DATASHEET, level_crossing, _color, _save
from chain_sim import run_points


def plot_per_chain_sir(sf_list, base, snr_offset, sirs, n, payload, jobs, seed,
                       use_cache, out_dir, show):
    pref = "" if base == "awgn" else base + "+"
    pts = [(f"{pref}tone:{s:g}", sf, round(DATASHEET[sf] + snr_offset, 2))
           for sf in sf_list for s in sirs]
    res = run_points(pts, n=n, payload=payload, seed=seed, jobs=jobs, use_cache=use_cache)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sir10 = {}
    for i, sf in enumerate(sf_list):
        rows = res[i * len(sirs):(i + 1) * len(sirs)]
        per = np.array([r["per"] for r in rows])
        ax.plot(sirs, per, "-o", ms=3, lw=1.7, color=_color(i), label=f"SF{sf}")
        ax.fill_between(sirs, [r["lo"] for r in rows], [r["hi"] for r in rows],
                        color=_color(i), alpha=0.13, lw=0)
        x10 = level_crossing(sirs, per, 0.1)
        sir10[sf] = x10
        if x10 is not None:
            ax.plot([x10], [0.1], "x", ms=9, mew=2, color=_color(i))
    ax.axhline(0.1, color="gray", ls=":", lw=1)
    ax.set_xlabel("SIR (сигнал / тон), дБ")
    ax.set_ylabel("PER")
    fad = {"awgn": "AWGN", "rayleigh": "AWGN + Рэлей", "rician": f"AWGN + Райс K={cfg.K_FACTOR_DB:g} дБ"}[base]
    ax.set_title(f"PER от уровня гармонической помехи ({fad})\n"
                 f"SNR = порог SF {snr_offset:+g} дБ, тон {cfg.HARMONIC_FREQ_OFFSET_HZ / 1e3:g} кГц от центра, "
                 f"гармоник: {cfg.HARMONIC_N}, payload={payload} байт, {n} пакетов на точку",
                 fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, ncol=2, loc="center left")
    _save(fig, f"PER_chain_vs_SIR_{base}.png", out_dir, show)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"PER_chain_vs_SIR_{base}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["scenario", "SF", "SNR_dB", "packets", "lost", "PER", "CI95_lo", "CI95_hi"])
        for r in res:
            w.writerow([r["scn"], r["sf"], f"{r['snr']:.2f}", r["n"], r["n_err"],
                        f"{r['per']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}"])
    print(f"[chain] CSV: {path}")
    return sir10


def main():
    ap = base_parser("PER от SIR гармонической помехи")
    ap.add_argument("--base", choices=["awgn", "rayleigh", "rician"], default="awgn")
    ap.add_argument("--snr-offset", type=float, default=3.0, help="SNR относительно порога SF, дБ")
    ap.add_argument("--sir-lo", type=float, default=-35.0)
    ap.add_argument("--sir-hi", type=float, default=0.0)
    ap.add_argument("--sir-step", type=float, default=2.5)
    a = ap.parse_args()
    sirs = [float(x) for x in np.arange(a.sir_lo, a.sir_hi + 1e-9, a.sir_step)]
    n = a.n_trials if a.n_trials != 300 else 200
    d = plot_per_chain_sir(a.sf, a.base, a.snr_offset, sirs, n, a.payload, a.jobs,
                           a.seed, not a.no_cache, a.out, a.show)
    print("\nSIR, при котором PER = 10 % (SIR ниже — пакеты теряются), дБ:")
    for sf in a.sf:
        print(f"  SF{sf}: {'—' if d[sf] is None else f'{d[sf]:6.1f}'}")


if __name__ == "__main__":
    main()
