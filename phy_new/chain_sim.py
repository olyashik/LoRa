# ---------------------------------------------------------------------------
# chain_sim.py — статистика PER по полной цепочке
#     передатчик (phy_transmit) → канал → приёмник (phy_receive)
# для набора SF и набора помех. Общий движок для plot_per_chain_*.py.
#
# Сценарий канала задаётся строкой:
#     "awgn"             только шум
#     "rician"           Райс (K = setup.K_FACTOR_DB) + шум
#     "rayleigh"         Рэлей + шум
#     "tone:-5"          гармоническая помеха (тон) с SIR = -5 дБ + шум
#     "rayleigh+tone:0"  можно комбинировать через «+»
#
# Пакет считается потерянным, если в нём есть хотя бы один битовый error.
# К PER добавляется 95 % доверительный интервал Уилсона.
#
# Канал собран из функций channel.py (add_rayleigh_fading, add_rician_fading,
# add_harmonic_interference), но шум и помеха нормируются на СРЕДНЮЮ мощность
# передатчика (до замирания и до помехи). В channel.add_awgn мощность шума
# считается по уже ослабленному/зашумлённому сигналу, из-за чего при блочном
# замирании Рэлея кривая PER почти совпадает с AWGN. Здесь SNR и SIR
# определены относительно мощности сигнала на выходе передатчика.
#
# Результаты кэшируются в extra_cache/chain_cache.json (по одной записи на точку).
# ---------------------------------------------------------------------------

import json
import math
import os
import zlib
from multiprocessing import Pool

import numpy as np

import setup as cfg
from phy import phy_transmit, phy_receive
from channel import (add_rayleigh_fading, add_rician_fading,
                     add_harmonic_interference)
from extra_common import make_params, DATASHEET, CACHE_DIR

CACHE_FILE = os.path.join(CACHE_DIR, "chain_cache.json")


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------

def parse_scenario(name):
    """'rayleigh+tone:-5' -> {'fading': 'rayleigh', 'sir': -5.0}"""
    sc = {"fading": None, "sir": None}
    for part in name.split("+"):
        part = part.strip().lower()
        if part == "awgn":
            continue
        if part in ("rayleigh", "rician"):
            sc["fading"] = part
        elif part.startswith("tone"):
            sc["sir"] = float(part.split(":")[1])
        else:
            raise ValueError(f"неизвестный сценарий: {part!r}")
    return sc


def scenario_label(name):
    sc = parse_scenario(name)
    parts = ["AWGN"]
    if sc["fading"] == "rayleigh":
        parts.append("Рэлей")
    elif sc["fading"] == "rician":
        parts.append(f"Райс K={cfg.K_FACTOR_DB:g} дБ")
    if sc["sir"] is not None:
        parts.append(f"тон SIR={sc['sir']:g} дБ")
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Канал
# ---------------------------------------------------------------------------

def apply_channel(tx, p, snr_db, sc, rng):
    """
    Один пакет через канал. SNR/SIR заданы относительно средней мощности tx.
    Порядок: замирание (мультипликативно) → тон (аддитивно) → шум.
    """
    fs = getattr(p, "fs", None) or p.bw
    p_ref = float(np.mean(np.abs(tx) ** 2))
    out = tx.copy()

    if sc["fading"] == "rician":
        out, _ = add_rician_fading(out, fs, cfg.K_FACTOR_DB, cfg.DOPPLER_HZ,
                                   seed=int(rng.integers(2 ** 31)))
    elif sc["fading"] == "rayleigh":
        out, _ = add_rayleigh_fading(out, fs, cfg.DOPPLER_HZ,
                                     seed=int(rng.integers(2 ** 31)))

    if sc["sir"] is not None:
        # помеха считается от незамирающего сигнала: вычитаем tx из tx + i(t)
        out = out + (add_harmonic_interference(
            tx, fs, sc["sir"], cfg.HARMONIC_FREQ_OFFSET_HZ,
            cfg.HARMONIC_N, cfg.HARMONIC_DECAY,
            seed=int(rng.integers(2 ** 31))) - tx)

    sigma = math.sqrt(p_ref / 10 ** (snr_db / 10))
    noise = (rng.standard_normal(len(out)) + 1j * rng.standard_normal(len(out))) \
        * sigma / math.sqrt(2)
    return out + noise


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """95 % доверительный интервал Уилсона для доли k/n."""
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _key(scn, sf, snr, n, payload, seed):
    return (f"{scn}|sf{sf}|snr{snr:.2f}|n{n}|pl{payload}|seed{seed}|cr{cfg.CR}|"
            f"bw{cfg.BW_KHZ}|k{cfg.K_FACTOR_DB}|dop{cfg.DOPPLER_HZ}|"
            f"off{cfg.HARMONIC_FREQ_OFFSET_HZ}|nh{cfg.HARMONIC_N}|dec{cfg.HARMONIC_DECAY}|"
            f"toa{int(cfg.ENABLE_TOA)}")


def _worker(job):
    """n пакетов в одной точке (сценарий, SF, SNR); выполняется в отдельном процессе."""
    key, scn, sf, snr, n, payload, seed = job
    sc = parse_scenario(scn)
    p = make_params(sf)
    ss = np.random.SeedSequence([seed, sf, int(round(snr * 100)) + 100000,
                                 zlib.crc32(scn.encode())])
    rng = np.random.default_rng(ss)
    n_err_pk, n_err_bits = 0, 0
    for _ in range(n):
        data = rng.bytes(payload)
        tx, _ = phy_transmit(data, p, enable_ToA=cfg.ENABLE_TOA)
        rx_sig = apply_channel(tx, p, snr, sc, rng)
        rx, _ = phy_receive(rx_sig, p, len(data), enable_ToA=cfg.ENABLE_TOA)
        rx = rx.ljust(len(data), b"\x00")[:len(data)]
        e = int(np.unpackbits(np.frombuffer(data, np.uint8) ^
                              np.frombuffer(rx, np.uint8)).sum())
        n_err_bits += e
        n_err_pk += 1 if e else 0
    return key, {"scn": scn, "sf": sf, "snr": snr, "n": n, "payload": payload,
                 "n_err": n_err_pk, "ber": n_err_bits / (8 * payload * n)}


def _load():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_FILE)


def run_points(points, n=100, payload=None, seed=12345, jobs=1, use_cache=True):
    """
    points: список (сценарий, sf, snr_db). Возвращает список записей
    {scn, sf, snr, n, n_err, per, lo, hi, ber} в том же порядке.
    """
    payload = cfg.PAYLOAD_BYTES if payload is None else payload
    cache = _load() if use_cache else {}
    keys = [_key(s, sf, snr, n, payload, seed) for s, sf, snr in points]
    todo = [(k, s, sf, snr, n, payload, seed)
            for k, (s, sf, snr) in zip(keys, points) if k not in cache]
    todo = list({t[0]: t for t in todo}.values())
    todo.sort(key=lambda t: -t[2])          # тяжёлые SF первыми
    if todo:
        print(f"[chain] новых точек: {len(todo)} из {len(points)}, "
              f"{n} пакетов на точку, процессов: {jobs}", flush=True)

        def consume(it):
            for i, (k, r) in enumerate(it, 1):
                cache[k] = r
                if i % 10 == 0 or i == len(todo):
                    _save_cache(cache)
                    print(f"[chain] {i}/{len(todo)}", flush=True)

        if jobs > 1 and len(todo) > 1:
            with Pool(min(jobs, len(todo))) as pool:
                consume(pool.imap_unordered(_worker, todo))
        else:
            consume(map(_worker, todo))
    else:
        print("[chain] все точки взяты из кэша", flush=True)

    out = []
    for k in keys:
        r = dict(cache[k])
        r["per"] = r["n_err"] / r["n"]
        r["lo"], r["hi"] = wilson(r["n_err"], r["n"])
        out.append(r)
    return out


def snr_grid(sf, lo=-3.0, hi=15.0, step=1.5):
    """Сетка SNR относительно паспортного порога SF: thr+lo ... thr+hi."""
    thr = DATASHEET[sf]
    return [round(thr + o, 2) for o in np.arange(lo, hi + 1e-9, step)]
