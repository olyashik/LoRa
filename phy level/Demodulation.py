# ---------------------------------------------------------------------------
# 7. Демодуляция: FFT-декодирование символов (приёмник)
# ---------------------------------------------------------------------------
# Источник: стандартный алгоритм de-chirping (Semtech patent EP2763321)
# Математика: Chiani & Elzanaty (IEEE TWC 2021), Section III

import numpy as np
import math
from typing import Tuple
import LoRa_Params
from Modulation import *
from ToA import *

def demodulate_symbol(rx_chirp: np.ndarray, p: LoRa_Params,
                      fs: float = None) -> int:
    '''
    Декодирует один символ из принятого чирпа.

    Алгоритм (de-chirping):
    1. Умножить принятый сигнал на сопряжённый базовый down-chirp:
           y(t) = r(t) * conj(down_chirp(t))
       После умножения чирп «распрямляется» в постоянный тон.

    2. Применить быстрое преобразование Фурье:
           Y[k] = FFT(y)
       Пик в БПФ соответствует стартовой частоте символа.

    3. Индекс пика = переданный символ:
           symbol = argmax(|Y[k]|²)  mod 2^SF

    При SNR < 0 дБ:
        Энергия символа «накапливается» в одном бине БПФ,
        тогда как шум распределён по всем бинам.
        Выигрыш: Усиление при обработке = 10·log10(2^SF) дБ
        (для SF=12: +36 дБ → работает при SNR = −20 дБ)
    '''
    M  = 2 ** p.sf
    if fs is None:
        fs = 8 * p.bw
    N  = int(round(M / p.bw * fs))  # сэмплов на символ

    if len(rx_chirp) < N:
        return 0

    rx = rx_chirp[:N]
    down = generate_chirp(0, p, fs, up=False)[:N]

    # De-chirping: умножение на сопряжённый down-chirp
    dechirped = rx * np.conj(down)

    # FFT и нахождение пика
    spectrum = np.abs(np.fft.fft(dechirped, n=M)) ** 2
    symbol   = int(np.argmax(spectrum)) % M

    return symbol


def symbols_to_bytes(symbols: np.ndarray, p: LoRa_Params) -> bytes:
    '''
    Обратное преобразование: символы → байты.
    Зеркальное отражение bytes_to_symbols().
    '''
    sf = p.sf
    bits = []
    for sym in symbols:
        b = gray_to_binary(sym)
        sym_bits = [(b >> (sf - 1 - i)) & 1 for i in range(sf)]
        bits.extend(sym_bits)

    bits = np.array(bits, dtype=np.uint8)

    # Обратное перемежение
    n_sym = len(symbols)
    if n_sym > 0:
        matrix = bits.reshape(sf, n_sym)
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


def demodulate(rx_signal: np.ndarray, p: LoRa_Params,
               n_payload_bytes: int, fs: float = None) -> Tuple[bytes, dict]:
    '''
    Полная демодуляция принятого сигнала.

    Возвращает:
        (данные: bytes, метаданные: dict)

    Метаданные (нужны MAC-уровню для ADR):
        snr_est   — оценка SNR из преамбулы [дБ]
        rssi_est  — оценка RSSI (условная) [дБм]
        toa_s     — реальный ToA [с]
        symbols   — число принятых символов
    '''
    if fs is None:
        fs = 8 * p.bw

    M = 2 ** p.sf
    N = int(round(M / p.bw * fs))  # сэмплов/символ

    # Пропускаем преамбулу и sync word
    skip_chirps = p.preamble_symbols + 2   # 2 = sync word
    offset = skip_chirps * N

    if len(rx_signal) <= offset:
        return b'', {"error": "signal too short"}

    payload_signal = rx_signal[offset:]

    # Оцениваем SNR по энергии преамбулы vs хвосту (грубо)
    preamble = rx_signal[:p.preamble_symbols * N]
    tail     = rx_signal[-N:] if len(rx_signal) > N else rx_signal
    snr_est  = round(10 * math.log10(
        max(np.mean(np.abs(preamble)**2) /
            max(np.mean(np.abs(tail)**2), 1e-12), 1e-6)), 1)

    # Декодируем символы
    symbols = []
    n_symbols = len(payload_signal) // N
    for i in range(n_symbols):
        chunk  = payload_signal[i*N:(i+1)*N]
        sym    = demodulate_symbol(chunk, p, fs)
        symbols.append(sym)

    symbols = np.array(symbols, dtype=np.int32)
    data    = symbols_to_bytes(symbols, p)[:n_payload_bytes]

    meta = {
        "snr_est_db": snr_est,
        "rssi_est_dbm": p.tx_power_dbm - 80,  # заглушка, в реале из RSSI-регистра
        "symbols_decoded": len(symbols),
        "toa": compute_toa(n_payload_bytes, p),
    }

    return data, meta
