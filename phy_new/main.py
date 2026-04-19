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

from LoRa_ToA import *
from params import LoRaParams
from modulate import *
from LoRa_ToA import *
from demodulate import *
from LoRa_Chirp import *
from LoRa_Coding import *
from channel import *
from BER_teor import *
from phy import *

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