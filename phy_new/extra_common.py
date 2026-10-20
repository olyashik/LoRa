# ---------------------------------------------------------------------------
# extra_common.py — общий код для графиков plot_*.py:
# параметры LoRa, расчёт ToA / goodput / RSSI / чувствительности,
# Монте-Карло PER(SNR) с кэшем (extra_cache/), оформление и сохранение.
# Сам по себе не запускается; используется из plot_per_snr.py, plot_per_payload.py,
# plot_goodput.py, plot_rssi.py, plot_per_distance.py.
# Требует файлы проекта из phy_new: setup.py, params.py, LoRa_ToA.py,
# channel.py, BER_teor.py, sim_core.py (положить extra_common.py рядом с ними).
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import setup as cfg
from params import LoRaParams
from LoRa_ToA import compute_toa
from channel import path_loss_db
from BER_teor import snr_threshold_db
from sim_core import compute_ber_per_curve

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "extra_cache")
DATASHEET = {sf: snr_threshold_db(LoRaParams(sf=sf)) for sf in range(7, 13)}
K_B = 1.38e-23


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def make_params(sf):
    """LoRaParams из setup.py (так же, как в BER.py и sim_core.py)."""
    p = LoRaParams(
        sf=sf, bw=cfg.BW_KHZ * 1e3, cr=cfg.CR,
        tx_power_dbm=cfg.TX_POWER_DBM, freq_hz=cfg.FREQ_MHZ * 1e6,
        preamble_symbols=cfg.PREAMBLE_SYMS, explicit_header=cfg.EXPLICIT_HDR,
    )
    if not cfg.ENABLE_LDR_AUTO:
        p.low_dr_opt = False
    return p


def shift_db(sf):
    """Сдвиг SNR → Eb/N0 (тот же, что в sim_core.py)."""
    rb = sf * (cfg.BW_KHZ * 1e3) / (2 ** sf) * (4 / (4 + cfg.CR))
    return 10 * math.log10((cfg.BW_KHZ * 1e3) / rb)


def default_window(sf, below=4.0, above=1.0):
    """Окно SNR для Монте-Карло вокруг паспортного порога SF."""
    return DATASHEET[sf] - below, DATASHEET[sf] + above


def level_crossing(x, y, level):
    """x, при котором y пересекает level (линейная интерполяция), иначе None."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i]))
    return None


def per_at_snr(curve, snr_db):
    """
    PER при произвольном SNR: интерполяция кривой Монте-Карло; левее окна
    PER = 1, правее окна PER = 0 (кривая монотонна, окно охватывает переход).
    """
    return np.interp(snr_db, curve["snr"], curve["per"], left=1.0, right=0.0)


def toa_s(sf, payload_bytes):
    return compute_toa(payload_bytes, make_params(sf))["toa_s"]


def goodput_bps(per, sf, payload_bytes):
    """Результирующая скорость, бит/с: доставленные биты нагрузки за время в эфире."""
    return 8 * payload_bytes * (1 - np.asarray(per, float)) / toa_s(sf, payload_bytes)


# --- энергетика линии связи (те же формулы, что в channel.simulate_channel) ---

def thermal_noise_dbm(bw_hz):
    return 10 * math.log10(K_B * cfg.TEMPERATURE_K * bw_hz) + 30


def rssi_dbm(distance_m, p, n=None):
    """Средний уровень сигнала на входе приёмника, дБм: P_tx − PL(d)."""
    n = cfg.PATH_LOSS_EXP if n is None else n
    d = np.atleast_1d(np.asarray(distance_m, float))
    pl = np.array([path_loss_db(x, p.freq_hz, n) for x in d])
    return p.tx_power_dbm - pl


def snr_at_distance(distance_m, p, n=None):
    """SNR на входе приёмника, дБ: RSSI − тепловой шум − коэффициент шума."""
    return rssi_dbm(distance_m, p, n) - thermal_noise_dbm(p.bw) - cfg.NOISE_FIGURE_DB


def sensitivity_dbm(sf, p):
    """Чувствительность по модели: kTB + NF + паспортный порог SNR."""
    return thermal_noise_dbm(p.bw) + cfg.NOISE_FIGURE_DB + DATASHEET[sf]


def distance_for_rssi(rssi_target_dbm, p, n=None):
    """Расстояние, на котором средний RSSI равен заданному уровню, м."""
    n = cfg.PATH_LOSS_EXP if n is None else n
    pl0 = path_loss_db(1.0, p.freq_hz, n)
    pl = p.tx_power_dbm - rssi_target_dbm
    return 10 ** ((pl - pl0) / (10 * n))


# ---------------------------------------------------------------------------
# Монте-Карло с кэшем
# ---------------------------------------------------------------------------

def _flags_key():
    return (f"ray{int(cfg.ENABLE_RAYLEIGH)}_ric{int(cfg.ENABLE_RICIAN)}_"
            f"k{cfg.K_FACTOR_DB}_dop{cfg.DOPPLER_HZ}_cr{cfg.CR}_bw{cfg.BW_KHZ}_"
            f"awgn{int(cfg.ENABLE_AWGN)}")


def _cache_path(tag, sf, n_trials, payload, lo, hi, step, seed):
    raw = f"{tag}|sf{sf}|n{n_trials}|pl{payload}|{lo:.3f}|{hi:.3f}|{step}|seed{seed}|{_flags_key()}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{tag}_sf{sf}_{h}.json")


def _worker(job):
    """Одна кривая PER(SNR) в окне [lo, hi]; выполняется в отдельном процессе."""
    sf, lo, hi, step, n_trials, payload, seed = job
    cfg.PAYLOAD_BYTES = payload
    cfg.RANDOM_SEED = seed
    np.random.seed(seed)
    sh = shift_db(sf)
    r = compute_ber_per_curve(sf, n_trials=n_trials,
                              eb_n0_min=lo + sh, eb_n0_max=hi + sh + 1e-6,
                              snr_step=step, verbose=False)
    return {"sf": sf, "payload": payload, "n_trials": n_trials,
            "snr": [float(x) for x in r["snr_range"]],
            "per": [float(x) for x in r["per"]],
            "ber": [float(x) for x in r["ber"]]}


def _run(jobs_list, n_jobs):
    keep = (cfg.PAYLOAD_BYTES, cfg.RANDOM_SEED)
    try:
        if n_jobs > 1 and len(jobs_list) > 1:
            with Pool(min(n_jobs, len(jobs_list))) as pool:
                return pool.map(_worker, jobs_list, chunksize=1)
        return [_worker(j) for j in jobs_list]
    finally:
        cfg.PAYLOAD_BYTES, cfg.RANDOM_SEED = keep


def get_curves(sf_list, n_trials=300, payload=None, step=0.5, window=None,
               jobs=1, seed=12345, use_cache=True, tag="per"):
    """
    Кривые PER(SNR) и BER(SNR) для списка SF.
    window(sf) -> (snr_lo, snr_hi); по умолчанию окно вокруг паспортного порога.
    Возвращает список словарей {sf, snr, per, ber, payload, n_trials}.
    """
    payload = cfg.PAYLOAD_BYTES if payload is None else payload
    window = window or default_window
    os.makedirs(CACHE_DIR, exist_ok=True)

    out, todo = {}, []
    for sf in sf_list:
        lo, hi = window(sf)
        path = _cache_path(tag, sf, n_trials, payload, lo, hi, step, seed)
        if use_cache and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out[sf] = json.load(f)
        else:
            todo.append((sf, lo, hi, step, n_trials, payload, seed, path))

    if todo:
        # тяжёлые SF первыми, чтобы пул не простаивал в конце
        todo.sort(key=lambda t: -t[0])
        print(f"[lora_extra] считаю {len(todo)} кривых PER(SNR), "
              f"{n_trials} пакетов на точку, процессов: {jobs}")
        res = _run([t[:7] for t in todo], jobs)
        for t, r in zip(todo, res):
            with open(t[7], "w", encoding="utf-8") as f:
                json.dump(r, f)
            out[t[0]] = r
    else:
        print("[lora_extra] кривые PER(SNR) взяты из кэша")

    for sf in sf_list:  # приводим к ndarray
        out[sf]["snr"] = np.array(out[sf]["snr"])
        out[sf]["per"] = np.array(out[sf]["per"])
        out[sf]["ber"] = np.array(out[sf]["ber"])
    return [out[sf] for sf in sf_list]


# ---------------------------------------------------------------------------
# Оформление
# ---------------------------------------------------------------------------

def _color(i):
    return plt.get_cmap("tab10")(i % 10)


def _subtitle(payload=None):
    s = f"BW={cfg.BW_KHZ} кГц, CR=4/{4 + cfg.CR}"
    if payload is not None:
        s += f", payload={payload} байт"
    if cfg.ENABLE_RICIAN:
        s += f", Райс K={cfg.K_FACTOR_DB} дБ"
    elif cfg.ENABLE_RAYLEIGH:
        s += ", Рэлей"
    else:
        s += ", AWGN"
    return s


def _save(fig, name, out_dir, show, tight=True):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"[lora_extra] сохранено: {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path




def base_parser(description):
    """Общие аргументы командной строки для всех plot_*.py."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--sf", nargs="+", type=int, default=[7, 8, 9, 10, 11, 12])
    ap.add_argument("--n-trials", type=int, default=300, help="пакетов на точку SNR")
    ap.add_argument("--snr-step", type=float, default=0.5)
    ap.add_argument("--payload", type=int, default=cfg.PAYLOAD_BYTES, help="байт")
    ap.add_argument("--out", default="figs_extra")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show", action="store_true")
    return ap
