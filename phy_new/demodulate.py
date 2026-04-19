# ---------------------------------------------------------------------------
# 6. Демодуляция: FFT-декодирование символов (приёмник)
# ---------------------------------------------------------------------------
# Источник: стандартный алгоритм de-chirping (Semtech patent EP2763321)
# Математика: Chiani & Elzanaty (IEEE TWC 2021), Section III

import numpy as np
from typing import Tuple
import math
from params import LoRaParams
from LoRa_Chirp import *
from LoRa_ToA import *
from LoRa_Coding import *
from gray import *

def demodulate_symbol(rx_chirp: np.ndarray, p: LoRaParams,
                      fs: float = None) -> int:
    """
    Декодирует один символ из принятого чирпа.

    Алгоритм де-чирпинга (Semtech patent EP2763321):
        1. base[n] = exp(j·2π·(c_0·n/BW + BW/(2T)·(n/BW)²))  — базовый UP-chirp (sym=0)
        2. y[n]    = rx[n] · conj(base[n])                      — де-чирпинг
        3. Y[k]    = FFT(y, n=M)                                — M-точечный спектр
        4. symbol  = argmax |Y[k]|²                             — пик = символ

    Почему умножаем на conj(UP-chirp), а не на DOWN-chirp:
        UP-chirp символа s:  phase_s = 2π·(c_s·t + BW/(2T)·t²)
        BASE UP-chirp (s=0): phase_0 = 2π·(c_0·t + BW/(2T)·t²)
        После умножения rx·conj(base):
            phase_diff = phase_s − phase_0 = 2π·(c_s − c_0)·t = 2π·(sym/M)·BW·t
        Это чистый тон с частотой sym·BW/M → FFT-пик точно на бине k = sym.

    Processing Gain: PG = 10·log10(2^SF) дБ
        SF=7:  PG ≈ 21 дБ  → работает при SNR = −7.5 дБ
        SF=12: PG ≈ 36 дБ  → работает при SNR = −20 дБ
    """
    M    = 2 ** p.sf
    N    = M                              # N = M (аналитический чирп)
    base = generate_chirp(0, p, up=True)  # базовый UP-chirp (sym=0)

    if len(rx_chirp) < N:
        return 0

    dechirped = rx_chirp[:N] * np.conj(base)
    spectrum  = np.abs(np.fft.fft(dechirped, n=M)) ** 2
    return int(np.argmax(spectrum)) % M



def symbols_to_bytes(symbols: np.ndarray, p: LoRaParams) -> bytes:
    """
    Обратное преобразование: символы → байты.
    Зеркальное отражение bytes_to_symbols().
    """
    sf   = p.sf
    bits = []
    for sym in symbols:
        sym = gray_decode(sym) 
        sym_bits = [(sym >> (sf - 1 - i)) & 1 for i in range(sf)]
        bits.extend(sym_bits)

    bits = np.array(bits, dtype=np.uint8)

    # Обратное перемежение
    n_sym = len(symbols)
    if n_sym > 0:
        matrix        = bits.reshape(sf, n_sym)
        deinterleaved = matrix.T.flatten()
    else:
        deinterleaved = bits
        
    # Декодирование FEC
    decoded = decode_cr(deinterleaved, p.cr)

    # Биты → байты
    pad = (-len(decoded)) % 8
    if pad:
        decoded = np.append(decoded, np.zeros(pad, dtype=np.uint8))
    return np.packbits(decoded).tobytes()

def demodulate(rx_signal: np.ndarray, p: LoRaParams,
               n_payload_bytes: int, fs: float = None) -> Tuple[bytes, dict]:
    """
    Полная демодуляция принятого сигнала.

    Возвращает:
        (данные: bytes, метаданные: dict)

    Метаданные (нужны MAC-уровню для ADR):
        snr_est   — оценка SNR из преамбулы [дБ]
        rssi_est  — оценка RSSI (условная) [дБм]
        toa       — ToA пакета
        symbols   — число принятых символов
    """

    M = 2 ** p.sf
    N = M  # сэмплов/символ

    # Пропускаем преамбулу и sync word
    skip_chirps = p.preamble_symbols + 2   # 2 = sync word
    offset      = skip_chirps * N
    
    if len(rx_signal) <= offset:
        return b'', {"error": "signal too short"}
    
    payload_signal = rx_signal[offset:]

    # Оцениваем SNR по энергии преамбулы vs хвосту
    preamble = rx_signal[:p.preamble_symbols * N]
    tail     = rx_signal[-N:] if len(rx_signal) > N else rx_signal
    snr_est  = round(10 * math.log10(
        max(np.mean(np.abs(preamble) ** 2) /
            max(np.mean(np.abs(tail) ** 2), 1e-12), 1e-6)), 1)

    # Декодируем символы
    n_symbols = len(payload_signal) // N
    symbols   = np.array([
        demodulate_symbol(payload_signal[i*N:(i+1)*N], p, fs)
        for i in range(n_symbols)
    ], dtype=np.int32)

    data = symbols_to_bytes(symbols, p)[:n_payload_bytes]
    meta = {
        "snr_est_db":      snr_est,
        "rssi_est_dbm":    p.tx_power_dbm - 80,
        "symbols_decoded": len(symbols),
        "toa":             compute_toa(n_payload_bytes, p),
    }
    return data, meta

