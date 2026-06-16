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


def apply_tx_power_and_freq_shift(signal: np.ndarray,
                                   pt_dbm: float,
                                   df: float,
                                   fs: float) -> np.ndarray:
    """
    Применяет мощность передатчика и частотный сдвиг к сигналу.
    Конвертация из MATLAB: 
        signal_mod = 10.^(Pt./20).*signal.*exp(-j.*2.*pi.*df/Fs.*(0:length(signal)-1))'

    Аргументы:
        signal  — комплексный I/Q сигнал единичной амплитуды
        pt_dbm  — мощность передатчика в дБм (например, 14.0)
        df      — частотное смещение в Гц (0 = идеальный канал без расстройки)
        fs      — частота дискретизации [Гц] (в модели LoRa fs = BW)

    Три операции (как в MATLAB):
        1. 10^(Pt/20)        — перевод дБ → линейная амплитуда
        2. * signal          — масштабирование сигнала по амплитуде
        3. * exp(-j*2π*df/fs*n) — комплексный частотный сдвиг на df Гц
                               при df=0 множитель равен 1 (нет эффекта)
    """
    # Амплитудный масштаб из мощности передатчика
    amplitude = 10 ** (pt_dbm / 20)

    # Временные индексы отсчётов: n = 0, 1, 2, ..., len(signal)-1
    n = np.arange(len(signal))

    # Комплексная экспонента для частотного сдвига
    # При df=0: exp(0) = 1, сигнал не изменяется
    freq_shift = np.exp(-1j * 2 * np.pi * df / fs * n)

    return amplitude * signal * freq_shift


def phy_transmit(data: bytes, p: LoRaParams,
                 fs: float = None, enable_ToA: bool = True) -> Tuple[np.ndarray, dict]:
    """
    MAC → PHY: байты → I/Q сигнал.

    Структура пакета:
        [преамбула: preamble_symbols × up-chirp'ов]
        [sync word: 2 × down-chirp'а]
        [полезная нагрузка: n_symbols × модулированных чирпов]

    После сборки сигнала применяется мощность передатчика и частотный сдвиг
    (функция apply_tx_power_and_freq_shift).
    """
    base  = generate_base_chirp(p)
    parts = []

    # Преамбула: preamble_symbols одинаковых up-chirp'ов (sym=0)
    for _ in range(p.preamble_symbols):
        parts.append(base.copy())

    # Sync word: два down-chirp'а — идентификатор сети LoRaWAN
    parts.append(generate_sync_word(p))

    # Полезная нагрузка: каждый символ → свой чирп
    symbols = modulate(data, p)
    for sym in symbols:
        parts.append(generate_chirp(sym, p))

    # Собираем все части в один непрерывный сигнал
    signal = np.concatenate(parts)

    # ── Применяем мощность передатчика и частотный сдвиг ─────────────────
    # fs = BW в аналитической модели (N = M = 2^SF отсчётов на символ)
    # df = 0: нет расстройки частоты между TX и RX
    # Эквивалент строки из MATLAB:
    #   signal_mod = 10.^(Pt./20).*signal.*exp(-j.*2.*pi.*df/Fs.*(0:N-1))'
    signal = apply_tx_power_and_freq_shift(
        signal,
        pt_dbm = p.tx_power_dbm,
        df     = 0,    # частотное смещение [Гц]
        fs     = p.bw  # fs = BW по построению модели
    )
    # ─────────────────────────────────────────────────────────────────────

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
        fs              — частота дискретизации (не используется, fs = BW)
        enable_ToA      — считать ли Time-on-Air и включать в метаданные

    Возвращает:
        (data: bytes, meta: dict)
        data — декодированные байты или b'' при ошибке
        meta — метаданные для MAC-уровня (SNR, RSSI, ToA, ...)
    """

    # M = 2^SF — число символов в алфавите LoRa и число отсчётов на символ
    # (fs = BW → N = M). Например: SF=9 → M=512, каждый символ кодирует 9 бит.
    M = 2 ** p.sf

    # ── 1. Вычисление offset полезной нагрузки ────────────────────────────
    #
    # Структура пакета (см. phy_transmit):
    #   [преамбула: preamble_symbols × M отсчётов]
    #   [sync word: 2 × M отсчётов]
    #   [полезная нагрузка: n_symbols × M отсчётов]
    #
    # Используем фиксированный offset (не корреляцию), потому что:
    #   - канал добавляет только шум, но не временной сдвиг;
    #   - корреляция при шуме даёт ложные пики → неверный offset;
    #   - длина преамбулы и sync word точно известна из параметров p.
    offset = (p.preamble_symbols + 2) * M  # +2 = два символа sync word

    # Защита: если сигнал короче преамбулы + sync word — декодировать нечего
    if len(rx_signal) <= offset:
        return b'', {"error": "signal too short"}

    # Вырезаем только полезную нагрузку — всё что после преамбулы и sync word
    payload_signal = rx_signal[offset:]

    # ── 2. Оценка SNR по преамбуле ────────────────────────────────────────
    #
    # Преамбула — одинаковые up-chirp'ы с постоянной мощностью сигнала.
    # Sync word — down-chirp'ы, при шуме менее коррелированы с базовым чирпом
    # и ведут себя ближе к шуму → используем как оценку уровня шума.
    #
    # SNR_est = 10 * log10(P_преамбула / P_sync)
    # Это грубая оценка, достаточная для ADR на MAC-уровне.
    preamble  = rx_signal[:p.preamble_symbols * M]          # только преамбула
    sync      = rx_signal[p.preamble_symbols * M : offset]  # только sync word
    pwr_sig   = np.mean(np.abs(preamble) ** 2)              # мощность сигнала
    pwr_noise = np.mean(np.abs(sync) ** 2)                  # мощность шума (≈)

    snr_est = round(10 * math.log10(
        max(pwr_sig / max(pwr_noise, 1e-12), 1e-6)  # защита от деления на ноль
    ), 1)

    # ── 3. Демодуляция символов ───────────────────────────────────────────
    #
    # Разбиваем payload_signal на блоки по M отсчётов — каждый блок один символ.
    # demodulate_symbol(): де-чирпинг (умножение на conj(base)) + argmax(FFT).
    # Остаток len(payload_signal) % M отсчётов отбрасывается — неполный символ
    # декодировать невозможно.
    n_symbols = len(payload_signal) // M

    if n_symbols == 0:
        return b'', {"error": "no symbols"}

    symbols = np.array([
        demodulate_symbol(payload_signal[i*M : (i+1)*M], p, fs)
        for i in range(n_symbols)
    ], dtype=np.int32)

    # ── 4. Символы → байты ────────────────────────────────────────────────
    #
    # demodulate() — обратная цепочка (зеркально modulate()):
    #   символы → gray_decode → биты → де-перемежение → decode_cr → байты
    #
    # Берём только первые n_payload_bytes байт: demodulate() может вернуть
    # чуть больше из-за padding'а при кодировании.
    data = demodulate(symbols, p)[:n_payload_bytes]

    # ── 5. Метаданные для MAC-уровня ──────────────────────────────────────
    #
    # MAC использует для:
    #   - ADR (Adaptive Data Rate): snr_est_db, rssi_est_dbm
    #   - планирования передачи: toa
    #   - отладки: symbols_decoded, sync_offset
    #
    # rssi_est_dbm — условная оценка: реального RSSI в симуляции нет,
    # используем tx_power минус условные потери 80 дБ.
    meta = {
        "snr_est_db":      snr_est,
        "rssi_est_dbm":    p.tx_power_dbm - 80,  # условная оценка
        "symbols_decoded": n_symbols,
        "sync_offset":     offset,                # фиксированный в симуляции
        "toa": compute_toa(n_payload_bytes, p) if enable_ToA else None,
    }

    return data, meta
