# ---------------------------------------------------------------------------
# Главный интерфейс PHY (используется MAC-уровнем)
# ---------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional
import math
from params import LoRaParams
from modulate import *
from LoRa_ToA import *
from LoRa_Chirp import *
from LoRa_Coding import *
from demodulate import *


def phy_transmit(data: bytes, p: LoRaParams,
                 fs: float = None, enable_ToA: bool = True) -> Tuple[np.ndarray, dict]:
    """
    MAC → PHY: байты → I/Q сигнал.
    Структура: [преамбула] [sync word] [символы данных]
    """
    base  = generate_base_chirp(p)
    parts = []

    for _ in range(p.preamble_symbols):
        parts.append(base.copy())

    parts.append(generate_sync_word(p))

    symbols = modulate(data, p)
    for sym in symbols:
        parts.append(generate_chirp(sym, p))

    signal = np.concatenate(parts)

    if enable_ToA:
        meta = compute_toa(len(data), p)
        meta.update({
            "n_payload_bytes": len(data),
            "sf":              p.sf,
            "bw_khz":          p.bw / 1e3,
            "cr":              f"4/{4 + p.cr}",
            "tx_power_dbm":    p.tx_power_dbm,
        })
    else:
        meta = None

    return signal, meta


def phy_receive(rx_signal: np.ndarray, p: LoRaParams,
                n_payload_bytes: int,
                fs: float = None, enable_ToA: bool = True) -> Tuple[Optional[bytes], dict]:
    """
    Точка входа PHY → MAC (приём).
    Принимает зашумлённый I/Q сигнал, возвращает декодированные байты.

    Аргументы:
        rx_signal       — принятый комплексный сигнал (I + jQ)
        p               — параметры LoRa (SF, BW, CR, ...)
        n_payload_bytes — ожидаемое число байт полезной нагрузки
        fs              — частота дискретизации (не используется, fs = BW по построению)
        enable_ToA      — считать ли Time-on-Air и включать в метаданные

    Возвращает:
        (data: bytes, meta: dict)
        data — декодированные байты или b'' при ошибке
        meta — метаданные для MAC-уровня (SNR, RSSI, ToA, ...)
    """

    # M = 2^SF — число символов в алфавите LoRa и одновременно
    # число отсчётов на один символ (так как fs = BW по построению модели).
    # Например: SF=9 → M=512, каждый символ кодирует 9 бит.
    M = 2 ** p.sf

    # ── 1. Вычисление offset полезной нагрузки ────────────────────────────
    #
    # Структура пакета (см. phy_transmit):
    #   [преамбула: preamble_symbols × M отсчётов]
    #   [sync word: 2 down-chirp'а × M отсчётов]
    #   [полезная нагрузка: n_symbols × M отсчётов]
    #
    # Offset — это граница между sync word и полезной нагрузкой.
    # Используем фиксированный offset (а не корреляцию), потому что:
    #   - в симуляции канал добавляет только шум, но не временной сдвиг;
    #   - корреляция при шуме даёт ложные пики и неверный offset;
    #   - длина преамбулы и sync word точно известна из параметров p.
    offset = (p.preamble_symbols + 2) * M  # +2 = два символа sync word

    # Защита от слишком короткого сигнала:
    # если сигнал короче преамбулы + sync word — декодировать нечего.
    if len(rx_signal) <= offset:
        return b'', {"error": "signal too short"}

    # Вырезаем только часть сигнала с полезной нагрузкой.
    # Всё что до offset — преамбула и sync word, они нам больше не нужны.
    payload_signal = rx_signal[offset:]

    # ── 2. Оценка SNR по преамбуле ────────────────────────────────────────
    #
    # Идея: преамбула состоит из одинаковых up-chirp'ов с постоянной мощностью.
    # Sync word — два down-chirp'а, которые при шуме менее коррелированы
    # с базовым чирпом и ведут себя ближе к шуму.
    #
    # SNR_est = 10 * log10(P_преамбула / P_sync)
    #
    # Это грубая оценка — она не учитывает реальную мощность шума,
    # но достаточна для ADR (Adaptive Data Rate) на MAC-уровне.
    preamble  = rx_signal[:p.preamble_symbols * M]          # только преамбула
    sync      = rx_signal[p.preamble_symbols * M : offset]  # только sync word
    pwr_sig   = np.mean(np.abs(preamble) ** 2)              # средняя мощность преамбулы
    pwr_noise = np.mean(np.abs(sync) ** 2)                  # средняя мощность sync (≈ шум)

    snr_est = round(10 * math.log10(
        max(pwr_sig / max(pwr_noise, 1e-12), 1e-6)  # защита от деления на ноль
    ), 1)

    # ── 3. Демодуляция символов ───────────────────────────────────────────
    #
    # Разбиваем payload_signal на блоки по M отсчётов — каждый блок один символ.
    # demodulate_symbol() выполняет де-чирпинг + FFT и возвращает номер символа (0..M-1).
    #
    # Остаток (len(payload_signal) % M отсчётов) отбрасывается —
    # неполный символ декодировать невозможно.
    n_symbols = len(payload_signal) // M

    if n_symbols == 0:
        return b'', {"error": "no symbols"}

    symbols = np.array([
        demodulate_symbol(payload_signal[i*M : (i+1)*M], p, fs)
        for i in range(n_symbols)
    ], dtype=np.int32)

    # ── 4. Символы → байты ────────────────────────────────────────────────
    #
    # demodulate() выполняет обратную цепочку (зеркально modulate()):
    #   символы → gray_decode → биты → де-перемежение → decode_cr → байты
    #
    # Берём только первые n_payload_bytes байт — demodulate() может вернуть
    # чуть больше из-за выравнивания (padding) при кодировании.
    data = demodulate(symbols, p)[:n_payload_bytes]

    # ── 5. Метаданные для MAC-уровня ──────────────────────────────────────
    #
    # MAC использует эти данные для:
    #   - ADR (Adaptive Data Rate): snr_est_db, rssi_est_dbm
    #   - планирования следующей передачи: toa
    #   - отладки: symbols_decoded, sync_offset
    #
    # rssi_est_dbm — условная оценка: в симуляции реального RSSI нет,
    # поэтому используем tx_power минус условные потери 80 дБ.
    meta = {
        "snr_est_db":      snr_est,
        "rssi_est_dbm":    p.tx_power_dbm - 80,  # условная оценка, не физическая
        "symbols_decoded": n_symbols,
        "sync_offset":     offset,                # всегда фиксированный в симуляции
        "toa": compute_toa(n_payload_bytes, p) if enable_ToA else None,
    }

    return data, meta
