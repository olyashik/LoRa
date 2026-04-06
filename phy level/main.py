'''
lora_phy.py — физический уровень LoRa на Python
================================================
Реализует модель передатчика и приёмника по формулам:
  - Semtech AN1200.22 (LoRa Modulation Basics)
  - Semtech SX1276 datasheet
  - Chiani & Elzanaty, IEEE TWC 2021 (математика CSS)
  - Nguyen et al., IEEE IoT Journal 2019 (эффективная CSS)

Интерфейс для MAC-уровня:
  tx_bytes = bytes([...])              # данные от MAC
  symbols  = phy_transmit(tx_bytes, params)   # PHY → символы
  rx_bytes, meta = phy_receive(symbols, params, snr_db)  # символы → байты
  meta содержит: rssi, snr, per, toa  # метаданные для MAC/ADR
'''

from LoRa_Params import LoRaParams
from ToA import *
from Transmitter import *
from FEC import *
from Modulation import *
from AWGN import *
from Demodulation import *
from BER import *
from PHY import *
import sys

# ---------------------------------------------------------------------------
# 10. Демонстрация
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
 
    print("(っ´ω`)ﾉ(╥ω╥) " * 7)
    print("LoRa PHY — демонстрация")
    print("(っ´ω`)ﾉ(╥ω╥) " * 7)
 
    # Параметры
    p = LoRaParams(sf=7, bw=125e3, cr=1, tx_power_dbm=14)
    payload = b"Hello LoRa!"
 
    # --- ToA ---
    toa = compute_toa(len(payload), p)
    print(f"\n[1] Time on Air для SF={p.sf}, BW={p.bw/1e3:.0f} кГц, "
          f"CR=4/{4+p.cr}, {len(payload)} байт:")
    print(f"    ToA       = {toa['toa_ms']} мс")
    print(f"    Преамбула = {toa['t_preamble_ms']} мс")
    print(f"    Данные    = {toa['t_payload_ms']} мс")
    print(f"    Rs        = {toa['symbol_rate_bps']} бит/с")
 
    # --- Передача через канал ---
    print(f"\n[2] Передача '{payload.decode()}'")
    tx_signal, tx_meta = phy_transmit(payload, p)
    print(f"    Длина сигнала: {len(tx_signal)} сэмплов")
 
    # Канал: 2 км, NF=6 дБ
    dist_m = 2000
    rx_signal, snr = simulate_channel(tx_signal, p, dist_m)
    print(f"    Расстояние:    {dist_m} м")
    print(f"    SNR на входе:  {snr:.1f} дБ")
    print(f"    Порог SNR:     {snr_threshold_db(p)} дБ")
    ok = snr >= snr_threshold_db(p)
    print(f"    Приём возможен: {'Да' if ok else 'Нет'}")
 
    # --- Демодуляция ---
    if ok:
        rx_data, meta = phy_receive(rx_signal, p, len(payload))
        print(f"\n[3] Принято: {rx_data}")
        print(f"    SNR (оценка приёмника): {meta['snr_est_db']} дБ")
 
    # --- BER по SNR ---
    print(f"\n[4] Теоретический BER:")
    for snr_test in [5, 0, -5, -7, -7.5, -10]:
        ber = ber_theory(snr_test, p)
        marker = " ← порог" if abs(snr_test - snr_threshold_db(p)) < 0.1 else ""
        print(f"    SNR={snr_test:5.1f} дБ  →  BER={ber:.2e}{marker}")
 
    # --- Сравнение SF ---
    print(f"\n[5] ToA vs SF (BW=125кГц, CR=4/5, 11 байт):")
    for sf in range(7, 13):
        pp  = LoRaParams(sf=sf, bw=125e3, cr=1)
        t   = compute_toa(11, pp)
        thr = snr_threshold_db(pp)
        print(f"    SF{sf}: ToA={t['toa_ms']:7.1f} мс, "
              f"Rs={t['symbol_rate_bps']:6.0f} бит/с, "
              f"SNR_min={thr:5.1f} дБ")
 
    print("\n" + "(; ω ; )ヾ(´∀`* ) " * 7)
    print("Интерфейс для MAC-уровня:")
    print("  signal, meta = phy_transmit(mac_frame_bytes, params)")
    print("  rx_bytes, meta = phy_receive(rx_signal, params, n_bytes)")
    print("  meta содержит: snr_est_db, rssi_est_dbm, toa, symbols_decoded")
    print("(; ω ; )ヾ(´∀`* ) " * 7)