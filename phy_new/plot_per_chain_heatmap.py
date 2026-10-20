# ---------------------------------------------------------------------------
# plot_per_chain_heatmap.py — сводная карта PER: строки — SF7..SF12,
# столбцы — сценарии помех, значение — PER при SNR = порог SF + offset
# (по умолчанию +3 дБ). В ячейке: PER и число потерянных пакетов / всего.
#
# Запуск из phy_new:
#   python plot_per_chain_heatmap.py
#   python plot_per_chain_heatmap.py --snr-offset 0 --scenarios awgn rayleigh tone:-10 tone:-20
# Если те же точки уже посчитаны plot_per_chain_snr.py (тот же --n-trials,
# --payload, offset кратен шагу 1,5 дБ от −3), они берутся из кэша.
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

import setup as cfg
from extra_common import base_parser, DATASHEET, _save
from chain_sim import run_points, scenario_label

DEFAULT_SCENARIOS = ["awgn", "rician", "rayleigh", "tone:-10", "tone:-20"]


def plot_heatmap(sf_list, scenarios, snr_offset, n, payload, jobs, seed, use_cache,
                 out_dir, show):
    pts = [(s, sf, round(DATASHEET[sf] + snr_offset, 2)) for sf in sf_list for s in scenarios]
    res = run_points(pts, n=n, payload=payload, seed=seed, jobs=jobs, use_cache=use_cache)
    m = np.array([r["per"] for r in res]).reshape(len(sf_list), len(scenarios))

    fig, ax = plt.subplots(figsize=(1.9 * len(scenarios) + 2.2, 0.62 * len(sf_list) + 2.6))
    im = ax.imshow(m, cmap="viridis_r", vmin=0, vmax=1, aspect="auto")
    for i in range(len(sf_list)):
        for j in range(len(scenarios)):
            r = res[i * len(scenarios) + j]
            ax.text(j, i, f"{r['per']:.2f}\n{r['n_err']}/{r['n']}", ha="center", va="center",
                    fontsize=8.5, color="white" if r["per"] > 0.45 else "black")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([scenario_label(s).replace(" + ", "\n+ ") for s in scenarios], fontsize=8)
    ax.set_yticks(range(len(sf_list)))
    ax.set_yticklabels([f"SF{sf}" for sf in sf_list])
    ax.set_title(f"PER при SNR = порог SF {snr_offset:+g} дБ, payload={payload} байт\n"
                 f"BW={cfg.BW_KHZ} кГц, CR=4/{4 + cfg.CR}; в ячейке: PER, потеряно/всего",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="PER")
    _save(fig, "PER_chain_heatmap.png", out_dir, show)
    return m


def main():
    ap = base_parser("Сводная карта PER: SF x помехи")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--snr-offset", type=float, default=3.0, help="SNR относительно порога SF, дБ")
    a = ap.parse_args()
    plot_heatmap(a.sf, a.scenarios, a.snr_offset, a.n_trials if a.n_trials != 300 else 200,
                 a.payload, a.jobs, a.seed, not a.no_cache, a.out, a.show)


if __name__ == "__main__":
    main()
