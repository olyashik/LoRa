"""
lora_phy.py — физический уровень LoRa на Python
================================================
Реализует модель передатчика и приёмника по формулам:
  - Semtech AN1200.22 (LoRa Modulation Basics)
  - Semtech SX1276 datasheet
  - Chiani & Elzanaty, IEEE TWC 2021 (математика CSS)
  - Nguyen et al., IEEE IoT Journal 2019 (эффективная CSS)

Зависимости:
  lora_params.py — класс LoRaParams (параметры трансивера)
  lora_setup.py  — константы симуляции (SF, BW, флаги канала и т.д.)

Интерфейс для MAC-уровня (Этап 2):
  signal, meta = phy_transmit(tx_bytes, params)
  rx_bytes, meta = phy_receive(signal, params, n_bytes)
  meta содержит: snr_est_db, rssi_est_dbm, toa, symbols_decoded
"""

import numpy as np
from typing import Tuple, Optional
import math

from params import LoRaParams


# ---------------------------------------------------------------------------
# 1. Time on Air (ToA)
# ---------------------------------------------------------------------------
# Источник: Semtech AN1200.22, раздел "LoRa Modem Time-on-Air"
# и SX1276 datasheet, формулы на стр. 31-32.

def compute_toa(payload_bytes: int, p: LoRaParams) -> dict:
    """
    Вычисляет Time on Air по формулам Semtech.

    Формула символьной скорости:
        Rs = BW / 2^SF   [символов/сек]

    Длительность одного символа:
        Ts = 1 / Rs = 2^SF / BW   [сек]

    Число символов преамбулы (с дробной частью 4.25):
        N_preamble = (preamble_symbols + 4.25) × Ts

    Число символов полезной нагрузки:
        N_payload = 8 + max(
            ceil( (8×PL - 4×SF + 28 + 16×CRC - 20×IH) /
                  (4×(SF - 2×LDR)) ) × (CR + 4),
            0 )
    где:
        PL  = число байт полезной нагрузки
        CRC = 1 (CRC включён, стандарт)
        IH  = 0 если явный заголовок (explicit header)
        LDR = 1 если Low Data Rate Optimize включён
    """
    Ts = (2 ** p.sf) / p.bw

    # Число символов полезной нагрузки
    crc  = 1                          # CRC всегда включён в LoRaWAN
    ih   = 0 if p.explicit_header else 1
    ldr  = 1 if p.low_dr_opt else 0

    numerator   = 8 * payload_bytes - 4 * p.sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (p.sf - 2 * ldr)
    payload_sym = 8 + max(math.ceil(numerator / denominator) * (p.cr + 4), 0)

    # Итоговый ToA
    t_preamble = (p.preamble_symbols + 4.25) * Ts
    t_payload  = payload_sym * Ts
    toa        = t_preamble + t_payload

    return {
        "toa_s":              round(toa, 6),
        "toa_ms":             round(toa * 1000, 3),
        "t_preamble_ms":      round(t_preamble * 1000, 3),
        "t_payload_ms":       round(t_payload * 1000, 3),
        "payload_symbols":    payload_sym,
        "symbol_duration_ms": round(Ts * 1000, 4),
        "symbol_rate_bps":    round(p.sf * p.bw / (2 ** p.sf), 2),
    }


# ---------------------------------------------------------------------------
# 2. Генерация чирп-сигнала (передатчик)
# ---------------------------------------------------------------------------
# Источник: Chiani & Elzanaty (IEEE TWC 2021), формула (1)-(3)
# Nguyen et al. (IEEE IoT Journal 2019), Section II

def generate_chirp(symbol_value: int, p: LoRaParams,
                   fs: float = None, up: bool = True) -> np.ndarray:
    """
    Генерирует один чирп-символ в комплексной форме I + jQ.

    Аналитическая модель (Semtech patent EP2763321):
        s[n] = exp(j·2π·(c_s·n/BW + sign·BW/(2T)·(n/BW)²))

    где:
        c_s  = (symbol_value/M − 0.5)·BW   — смещение несущей символа [Гц]
        M    = 2^SF                          — число символов
        T    = M/BW                          — длительность символа [с]
        n    = 0, 1, …, M−1                 — дискретный индекс отсчёта
        sign = +1 (up-chirp) или −1 (down-chirp)

    Дискретизация: fs = BW → N = M отсчётов на символ.
    При этом после де-чирпинга FFT-бин k точно равен symbol_value.

    Почему не cumsum + % BW:
        Оборачивание частоты через % вносит фазовые разрывы в сигнал.
        De-chirping (FFT) требует непрерывной фазы — иначе пик размывается
        и символ декодируется неверно даже без шума.
        Аналитическая квадратичная фаза разрывов не имеет.

    |s[n]| = 1 — константная огибающая (важно для усилителя мощности).

    Аргументы:
        symbol_value: кодируемый символ (0 .. 2^SF - 1)
        p:  параметры LoRa
        fs: не используется (оставлен для совместимости сигнатуры)
        up: True = up-chirp (данные), False = down-chirp (для демодуляции)
    """
    M    = 2 ** p.sf
    T    = M / p.bw
    N    = M                          # fs = BW → N = M отсчётов на символ
    t    = np.arange(N) / p.bw       # временная ось [с]
    c_s  = (symbol_value / M - 0.5) * p.bw   # смещение несущей [Гц]
    sign = 1 if up else -1
    phase = 2 * np.pi * (c_s * t + sign * p.bw / (2 * T) * t ** 2)
    return np.exp(1j * phase)        # комплексный сигнал единичной амплитуды


def generate_base_chirp(p: LoRaParams, fs: float = None) -> np.ndarray:
    """Базовый up-chirp (symbol=0) — используется для преамбулы."""
    return generate_chirp(0, p, up=True)


def generate_sync_word(p: LoRaParams, fs: float = None) -> np.ndarray:
    """
    Sync word LoRaWAN: символы 0x34 (два даун-чирпа + 2.25 up-чирпа).
    Упрощённая версия: два down-chirp подряд.
    В реальном SX1276 sync word = 0x12 (private) или 0x34 (public/LoRaWAN).
    """
    down = generate_chirp(0, p, up=False)
    return np.concatenate([down, down])


# ---------------------------------------------------------------------------
# 3. Кодирование Хэмминга (FEC / Coding Rate)
# ---------------------------------------------------------------------------
# Источник: Semtech SX1276 datasheet, раздел "Error Coding"
# CR=1 → 4/5 (1 бит коррекции на 4 бита данных)
# CR=4 → 4/8 = ½ (4 бита коррекции на 4 бита данных, максимальная защита)

def encode_cr(bits: np.ndarray, cr: int) -> np.ndarray:
    """
    Упрощённое кодирование с избыточностью согласно Coding Rate.
    Реальный LoRa использует укороченный код Хэмминга (4, 4+cr).

    CR=1: каждые 4 бита → 5 бит (1 бит чётности)
    CR=2: каждые 4 бита → 6 бит (2 бита чётности)
    CR=3: каждые 4 бита → 7 бит (3 бита чётности)
    CR=4: каждые 4 бита → 8 бит (4 бита чётности = повторение)
    """
    out = []
    for i in range(0, len(bits) - len(bits) % 4, 4):
        nibble = bits[i:i+4]
        parity = []
        if cr >= 1: parity.append(nibble[0] ^ nibble[1] ^ nibble[2])
        if cr >= 2: parity.append(nibble[1] ^ nibble[2] ^ nibble[3])
        if cr >= 3: parity.append(nibble[0] ^ nibble[2] ^ nibble[3])
        if cr >= 4: parity.append(nibble[0] ^ nibble[1] ^ nibble[3])
        out.extend(nibble)
        out.extend(parity)
    return np.array(out, dtype=np.uint8)


def decode_cr(bits: np.ndarray, cr: int) -> np.ndarray:
    """Декодирование: берём только информационные биты (первые 4 из кодового слова)."""
    codeword_len = 4 + cr
    out = []
    for i in range(0, len(bits) - len(bits) % codeword_len, codeword_len):
        out.extend(bits[i:i+4])
    return np.array(out, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 4. Модуляция: байты → символы → чирпы (передатчик)
# ---------------------------------------------------------------------------

def bytes_to_symbols(data: bytes, p: LoRaParams) -> np.ndarray:
    """
    Преобразует байты в LoRa-символы.

    Шаги:
    1. Байты → биты
    2. Кодирование FEC (Coding Rate)
    3. Перемежение (interleaving) — защита от пакетных ошибок
    4. Биты → символы (SF бит на символ)

    Число символов = ceil(len(encoded_bits) / SF)
    """
    # Байты → биты (MSB первый)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # FEC кодирование
    encoded = encode_cr(bits, p.cr)

    # Простое перемежение: читаем биты по диагонали (SF×SF матрица)
    sf  = p.sf
    pad = (-len(encoded)) % sf
    if pad:
        encoded = np.append(encoded, np.zeros(pad, dtype=np.uint8))
    matrix      = encoded.reshape(-1, sf)
    interleaved = matrix.T.flatten()

    # Биты → символы (каждые SF бит = один символ)
    symbols = []
    for i in range(0, len(interleaved), sf):
        chunk = interleaved[i:i+sf]
        if len(chunk) == sf:
            sym = int(''.join(map(str, chunk)), 2) % (2 ** sf)
            symbols.append(sym)

    return np.array(symbols, dtype=np.int32)


def modulate(data: bytes, p: LoRaParams, fs: float = None) -> np.ndarray:
    """
    Полная модуляция: байты → радиосигнал I+jQ.

    Структура пакета (физический фрейм):
        [преамбула] [sync word] [заголовок] [полезная нагрузка]

    Преамбула: N up-chirp'ов (N = preamble_symbols, стандарт = 8)
    Sync word: 2 down-chirp'а (идентификатор сети LoRaWAN)
    Далее: символы данных
    """
    base  = generate_base_chirp(p)
    parts = []

    # Преамбула: preamble_symbols up-chirp'ов
    for _ in range(p.preamble_symbols):
        parts.append(base.copy())

    # Sync word (2 down-chirp)
    parts.append(generate_sync_word(p))

    # Символы данных
    symbols = bytes_to_symbols(data, p)
    for sym in symbols:
        parts.append(generate_chirp(sym, p))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 5. AWGN-канал + затухание
# ---------------------------------------------------------------------------
# Источник: формула Фрииса + логарифмическая модель потерь (log-distance)

def path_loss_db(distance_m: float, freq_hz: float,
                 path_loss_exp: float = 2.7) -> float:
    """
    Логарифмическая модель потерь.

    Фриис (свободное пространство, n=2):
        PL_0 = 20·log10(4π·f/c)   [дБ на 1 м]

    Общая модель:
        PL(d) = PL_0 + 10·n·log10(d)
    """
    c    = 3e8
    pl_0 = 20 * math.log10(4 * math.pi * freq_hz / c)
    return pl_0 + 10 * path_loss_exp * math.log10(max(distance_m, 1))


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Добавляет комплексный белый гауссовский шум (AWGN).

    SNR = P_signal / P_noise = 10^(snr_db/10)
    σ² = P_signal / SNR        (дисперсия шума)
    σ  = sqrt(P_signal / SNR)  (среднеквадратическое отклонение)

    Шум комплексный: n = (n_I + j·n_Q) / √2, оба компонента N(0,σ²/2)
    """
    p_signal = np.mean(np.abs(signal) ** 2)
    snr_lin  = 10 ** (snr_db / 10)
    sigma    = math.sqrt(p_signal / snr_lin)
    noise    = (np.random.randn(len(signal)) +
                1j * np.random.randn(len(signal))) * sigma / math.sqrt(2)
    return signal + noise


def simulate_channel(signal: np.ndarray, p: LoRaParams,
                     distance_m: float,
                     noise_figure_db: float = 6.0,
                     temperature_k: float = 290.0,
                     path_loss_exp: float = 2.7,
                     enable_awgn: bool = True,
                     enable_path_loss: bool = True,
                     fixed_snr_db: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Модель канала с управляемыми флагами.

    Флаги из lora_setup.py:
        enable_path_loss = True  → потери считаются по расстоянию и модели Фрииса
        enable_path_loss = False → используется fixed_snr_db напрямую
        enable_awgn      = True  → добавляется AWGN-шум
        enable_awgn      = False → сигнал проходит без изменений (идеальный канал)

    Тепловой шум (Джонсон–Найквист):
        N_thermal = k·T·B  [Вт]
        N_dBm = 10·log10(k·T·B) + 30

    SNR на входе приёмника (при enable_path_loss = True):
        SNR = P_tx - PL(d) - N_dBm - NF
    """
    if not enable_path_loss:
        # Канал без затухания: используем фиксированный SNR из lora_setup
        snr_db = fixed_snr_db
    else:
        # Полная физическая модель
        k_b           = 1.38e-23
        n_thermal_dbm = 10 * math.log10(k_b * temperature_k * p.bw) + 30
        pl            = path_loss_db(distance_m, p.freq_hz, path_loss_exp)
        p_rx_dbm      = p.tx_power_dbm - pl
        snr_db        = p_rx_dbm - n_thermal_dbm - noise_figure_db

    if not enable_awgn:
        # Идеальный канал: возвращаем сигнал без шума
        return signal.copy(), snr_db

    return add_awgn(signal, snr_db), snr_db


# ---------------------------------------------------------------------------
# 6. Демодуляция: FFT-декодирование символов (приёмник)
# ---------------------------------------------------------------------------
# Источник: стандартный алгоритм de-chirping (Semtech patent EP2763321)
# Математика: Chiani & Elzanaty (IEEE TWC 2021), Section III

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
    N = M                              # N = M (аналитический чирп, fs = BW)

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
        demodulate_symbol(payload_signal[i*N:(i+1)*N], p)
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


# ---------------------------------------------------------------------------
# 7. BER: теоретическая кривая
# ---------------------------------------------------------------------------
# Источник: Elshabrawy & Robert, IEEE Comm. Letters 2018

def ber_theory(snr_db: float, p: LoRaParams) -> float:
    """
    Теоретический BER для LoRa (некогерентный приёмник) в канале AWGN.

    Приближение (Elshabrawy & Robert 2018):
        BER ≈ (2^SF / 2) / (2^SF - 1) × erfc(√(SF·SNR / 2))
    """
    from scipy.special import erfc
    snr_lin = 10 ** (snr_db / 10)
    M       = 2 ** p.sf
    ber     = ((M / 2) / (M - 1)) * erfc(math.sqrt(p.sf * snr_lin / 2))
    return min(max(ber, 1e-10), 0.5)


def snr_threshold_db(p: LoRaParams) -> float:
    """
    Пороговый SNR для приёма (из datasheet SX1276, Table 13).
    При этом SNR: BER ≈ 1% (стандарт Semtech).

    SF7:  −7.5 дБ   SF10: −15.0 дБ
    SF8:  −10.0 дБ  SF11: −17.5 дБ
    SF9:  −12.5 дБ  SF12: −20.0 дБ
    """
    return {7: -7.5, 8: -10.0, 9: -12.5,
            10: -15.0, 11: -17.5, 12: -20.0}[p.sf]


# ---------------------------------------------------------------------------
# 8. Главный интерфейс PHY (используется MAC-уровнем)
# ---------------------------------------------------------------------------

def phy_transmit(data: bytes, p: LoRaParams,
                 fs: float = None) -> Tuple[np.ndarray, dict]:
    """
    Точка входа MAC → PHY (передача).
    MAC передаёт сырые байты (PHYPayload), PHY возвращает I/Q сигнал и метаданные.
    """
    signal = modulate(data, p, fs)
    meta   = compute_toa(len(data), p)
    meta.update({
        "n_payload_bytes": len(data),
        "sf":              p.sf,
        "bw_khz":          p.bw / 1e3,
        "cr":              f"4/{4 + p.cr}",
        "tx_power_dbm":    p.tx_power_dbm,
    })
    return signal, meta


def phy_receive(rx_signal: np.ndarray, p: LoRaParams,
                n_payload_bytes: int,
                fs: float = None) -> Tuple[Optional[bytes], dict]:
    """
    Точка входа PHY → MAC (приём).
    PHY принимает I/Q сигнал, возвращает байты и метаданные для ADR.
    """
    return demodulate(rx_signal, p, n_payload_bytes, fs)


# ---------------------------------------------------------------------------
# 9. Демонстрация
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import setup as cfg

    # Собираем LoRaParams из констант lora_setup.py
    p = LoRaParams(
        sf               = cfg.SF,
        bw               = cfg.BW_KHZ * 1e3,
        cr               = cfg.CR,
        tx_power_dbm     = cfg.TX_POWER_DBM,
        freq_hz          = cfg.FREQ_MHZ * 1e6,
        preamble_symbols = cfg.PREAMBLE_SYMS,
        explicit_header  = cfg.EXPLICIT_HDR,
    )
    if not cfg.ENABLE_LDR_AUTO:
        p.low_dr_opt = False

    SEP = "=" * 60
    print(SEP)
    print("LoRa PHY — демонстрация")
    print(SEP)

    # Конфигурация
    print(f"\n[Конфигурация]")
    print(f"  SF={p.sf}  BW={p.bw/1e3:.0f} кГц  CR=4/{4+p.cr}"
          f"  P_tx={p.tx_power_dbm} дБм  f={p.freq_hz/1e6:.1f} МГц")
    print(f"  LDR Optimize : {'включён' if p.low_dr_opt else 'выключен'}")
    print(f"  AWGN         : {'включён' if cfg.ENABLE_AWGN else 'ВЫКЛЮЧЕН'}")
    if cfg.ENABLE_PATH_LOSS:
        print(f"  Path loss    : включён"
              f"  (d={cfg.DISTANCE_M} м, n={cfg.PATH_LOSS_EXP},"
              f" NF={cfg.NOISE_FIGURE_DB} дБ)")
    else:
        print(f"  Path loss    : ВЫКЛЮЧЕН → фикс. SNR = {cfg.FIXED_SNR_DB} дБ")

    # --- Генерация случайных бит ---
    # PAYLOAD_BYTES из lora_setup задаёт размер пакета;
    # из него получаем число бит и упаковываем в байты для передачи через PHY.
    n_bits   = cfg.PAYLOAD_BYTES * 8
    rng      = np.random.default_rng(cfg.RANDOM_SEED)
    tx_bits  = rng.integers(0, 2, n_bits, dtype=np.uint8)
    # Упаковываем биты в байты для phy_transmit (PHY работает с байтами)
    tx_bytes = np.packbits(tx_bits).tobytes()

    # --- ToA ---
    toa = compute_toa(len(tx_bytes), p)
    print(f"\n[1] Time on Air  (SF={p.sf}, BW={p.bw/1e3:.0f} кГц,"
          f" CR=4/{4+p.cr}, {len(tx_bytes)} байт = {n_bits} бит)")
    print(f"    ToA       = {toa['toa_ms']} мс")
    print(f"    Преамбула = {toa['t_preamble_ms']} мс")
    print(f"    Данные    = {toa['t_payload_ms']} мс")
    print(f"    Rs        = {toa['symbol_rate_bps']} бит/с")

    # --- Передача ---
    print(f"\n[2] Передача  ({n_bits} случайных бит)")
    print(f"    TX биты: {''.join(map(str, tx_bits[:32].tolist()))}...  "
          f"(первые 32 из {n_bits})")
    tx_signal, tx_meta = phy_transmit(tx_bytes, p)
    print(f"    Длина сигнала : {len(tx_signal)} сэмплов")

    # --- Канал ---
    rx_signal, snr = simulate_channel(
        tx_signal, p,
        distance_m      = cfg.DISTANCE_M,
        noise_figure_db = cfg.NOISE_FIGURE_DB,
        temperature_k   = cfg.TEMPERATURE_K,
        path_loss_exp   = cfg.PATH_LOSS_EXP,
        enable_awgn     = cfg.ENABLE_AWGN,
        enable_path_loss= cfg.ENABLE_PATH_LOSS,
        fixed_snr_db    = cfg.FIXED_SNR_DB,
    )
    threshold = snr_threshold_db(p)
    ok        = snr >= threshold
    print(f"\n[3] Канал")
    print(f"    Расстояние    : {cfg.DISTANCE_M} м")
    print(f"    SNR на входе  : {snr:.1f} дБ")
    print(f"    Порог SNR     : {threshold} дБ")
    print(f"    Приём возможен: {'ДА' if ok else 'НЕТ — ожидаются ошибки'}")

    # --- Приём и разбор бит ---
    rx_data, meta = phy_receive(rx_signal, p, len(tx_bytes))
    # Распаковываем байты обратно в биты
    rx_bytes_padded = rx_data.ljust(len(tx_bytes), b'\x00')
    rx_bits = np.unpackbits(
        np.frombuffer(rx_bytes_padded, dtype=np.uint8)
    )[:n_bits]

    print(f"\n[4] Приём  ({n_bits} бит)")
    print(f"    RX биты: {''.join(map(str, rx_bits[:32].tolist()))}...  "
          f"(первые 32 из {n_bits})")
    print(f"    SNR (оценка приёмника): {meta['snr_est_db']} дБ")

    # --- Анализ ошибок ---
    error_mask = tx_bits ^ rx_bits                   # 1 там, где бит не совпал
    n_errors   = int(np.sum(error_mask))
    ber_sim    = n_errors / n_bits
    error_pos  = [int(i) for i in np.where(error_mask == 1)[0]]
    packet_ok  = n_errors == 0

    print(f"\n[5] Анализ ошибок")
    print(f"    Бит передано   : {n_bits}")
    print(f"    Бит ошибочных  : {n_errors}")
    print(f"    BER            : {ber_sim:.4f}  ({ber_sim*100:.2f}%)")
    print(f"    Пакет принят   : {'ДА ✓' if packet_ok else 'НЕТ ✗'}")
    if error_pos:
        shown = error_pos[:20]
        tail  = f"... (всего {len(error_pos)})" if len(error_pos) > 20 else ""
        print(f"    Позиции ошибок : {shown}{tail}")
    else:
        print(f"    Позиции ошибок : нет")

    # --- Карта ошибочных бит ---
    width = 64
    print(f"\n[6] Карта бит  ('.' верно  'X' ошибка  по {width} бит/строка)")
    chars = ['X' if b else '.' for b in error_mask]
    for i in range(0, len(chars), width):
        row = chars[i: i + width]
        n_err_row = row.count('X')
        print(f"    {''.join(row)}  [{n_err_row:3d} ош.]")

    # --- Теоретический BER ---
    print(f"\n[7] Теоретический BER:")
    for snr_test in [5, 0, -5, -7, -7.5, -10]:
        ber    = ber_theory(snr_test, p)
        marker = " ← порог" if abs(snr_test - threshold) < 0.1 else ""
        print(f"    SNR={snr_test:5.1f} дБ  →  BER={ber:.2e}{marker}")

    # --- Сравнение SF ---
    print(f"\n[8] ToA vs SF  (BW=125 кГц, CR=4/5, {len(tx_bytes)} байт):")
    for sf in range(7, 13):
        pp  = LoRaParams(sf=sf, bw=125e3, cr=1)
        t   = compute_toa(len(tx_bytes), pp)
        thr = snr_threshold_db(pp)
        print(f"    SF{sf}: ToA={t['toa_ms']:7.1f} мс,"
              f" Rs={t['symbol_rate_bps']:6.0f} бит/с,"
              f" SNR_min={thr:5.1f} дБ")

    print(f"\n{SEP}")
    print("Интерфейс для MAC-уровня:")
    print("  signal, meta = phy_transmit(mac_frame_bytes, params)")
    print("  rx_bytes, meta = phy_receive(rx_signal, params, n_bytes)")
    print("  meta: snr_est_db, rssi_est_dbm, toa, symbols_decoded")
    print(SEP)