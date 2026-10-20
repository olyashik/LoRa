# ---------------------------------------------------------------------------
# plot_per_chain_snr.py — PER(SNR) по полной цепочке TX → канал → RX
# для SF7..SF12 (по панели на каждый SF) и набора помех/замираний.
#
# Каждая точка — N пакетов, закрашенная полоса — 95 % интервал Уилсона.
# Вертикальный пунктир — паспортный порог SX1276 для этого SF.
# Рядом сохраняется CSV со всеми числами (для таблицы в статье).
#
# Запуск из phy_new (рядом chain_sim.py, extra_common.py):
#   python plot_per_chain_snr.py
#   python plot_per_chain_snr.py --scenarios awgn rayleigh tone:-15 --n-trials 300
#   python plot_per_chain_snr.py --sf 7 12 --n-trials 50      # быстрый черновик
# Сценарии: awgn, rician, rayleigh, tone:<SIR дБ>, комбинации через «+»
# (например rayleigh+tone:-10). Параметры замираний и тона — в setup.py.
# ---------------------------------------------------------------------------

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import setup as cfg
from extra_common import base_parser, DATASHEET, _color, _save
from chain_sim import run_points, snr_grid, scenario_label

DEFAULT_SCENARIOS = ["awgn", "rician", "rayleigh", "tone:-10", "tone:-20"]


def plot_per_chain_snr(sf_list, scenarios, n, payload, step, lo, hi, jobs, seed,
                       use_cache, out_dir, show):
    pts = [(s, sf, snr) for sf in sf_list for s in scenarios
           for snr in snr_grid(sf, lo, hi, step)]
    res = run_points(pts, n=n, payload=payload, seed=seed, jobs=jobs, use_cache=use_cache)
    by = {}
    for r in res:
        by.setdefault((r["sf"], r["scn"]), []).append(r)

    ncol = 3 if len(sf_list) > 3 else len(sf_list)
    nrow = int(np.ceil(len(sf_list) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.0 * nrow + 0.9),
                             squeeze=False, sharey=True)
    for ax, sf in zip(axes.ravel(), sf_list):
        for i, s in enumerate(scenarios):
            rows = sorted(by[(sf, s)], key=lambda r: r["snr"])
            x = [r["snr"] for r in rows]
            ax.plot(x, [r["per"] for r in rows], "-o", ms=3, lw=1.6, color=_color(i),
                    label=scenario_label(s))
            ax.fill_between(x, [r["lo"] for r in rows], [r["hi"] for r in rows],
                            color=_color(i), alpha=0.15, lw=0)
        ax.axvline(DATASHEET[sf], color="gray", ls="--", lw=1)
        ax.set_title(f"SF{sf}", fontsize=11)
        ax.set_xlabel("SNR, дБ")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.35)
    for ax in axes[:, 0]:
        ax.set_ylabel("PER")
    for ax in axes.ravel()[len(sf_list):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), fontsize=9, frameon=False)
    fig.suptitle(f"PER по цепочке TX → канал → RX, BW={cfg.BW_KHZ} кГц, CR=4/{4 + cfg.CR}, "
                 f"payload={payload} байт, {n} пакетов на точку\n"
                 f"пунктир — паспортный порог SX1276, полоса — 95 % интервал Уилсона",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.045 if len(l) <= 5 else 0.09, 1, 0.93))
    _save(fig, "PER_chain_vs_SNR.png", out_dir, show, tight=False)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "PER_chain_vs_SNR.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["scenario", "SF", "SNR_dB", "SNR_minus_thr_dB", "packets", "lost",
                    "PER", "CI95_lo", "CI95_hi", "BER"])
        for r in res:
            w.writerow([r["scn"], r["sf"], f"{r['snr']:.2f}",
                        f"{r['snr'] - DATASHEET[r['sf']]:.1f}", r["n"], r["n_err"],
                        f"{r['per']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                        f"{r['ber']:.3e}"])
    print(f"[chain] CSV: {path}")
    return res


def main():
    ap = base_parser("PER(SNR) по цепочке TX-канал-RX для набора помех")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--lo", type=float, default=-3.0, help="начало окна SNR относительно порога, дБ")
    ap.add_argument("--hi", type=float, default=15.0, help="конец окна SNR относительно порога, дБ")
    a = ap.parse_args()
    step = a.snr_step if a.snr_step != 0.5 else 1.5     # шаг по умолчанию для этого графика
    res = plot_per_chain_snr(a.sf, a.scenarios, a.n_trials if a.n_trials != 300 else 200,
                             a.payload, step, a.lo, a.hi, a.jobs, a.seed,
                             not a.no_cache, a.out, a.show)
    # SNR (над порогом), при котором PER ≤ 10 %
    print("\nМинимальный SNR - порог, дБ, при котором PER <= 10 % (— : не достигается в окне):")
    for sf in a.sf:
        row = []
        for s in a.scenarios:
            ok = [r["snr"] - DATASHEET[sf] for r in res
                  if r["sf"] == sf and r["scn"] == s and r["per"] <= 0.1]
            row.append(f"{min(ok):+5.1f}" if ok else "   — ")
        print(f"  SF{sf}: " + " | ".join(row))
    print("  столбцы: " + " | ".join(a.scenarios))


if __name__ == "__main__":
    main()
