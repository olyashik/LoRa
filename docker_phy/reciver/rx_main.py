import numpy as np
import socket
import setup as cfg
from params import LoRaParams
from phy import phy_receive
from transport import UDPFrameReceiver, unpack_frame

LISTEN_HOST = "0.0.0.0"  # Слушаем все интерфейсы
LISTEN_PORT = 5005

print(f"[RX] Запуск UDP приемника на {LISTEN_HOST}:{LISTEN_PORT}")

# Проверяем доступность порта
try:
    test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    test_sock.bind((LISTEN_HOST, LISTEN_PORT))
    test_sock.close()
    print(f"[RX] Порт {LISTEN_PORT} доступен")
except Exception as e:
    print(f"[RX] WARNING: Порт {LISTEN_PORT} занят: {e}")

p = LoRaParams(
    sf=cfg.SF, bw=cfg.BW_KHZ * 1e3, cr=cfg.CR,
    tx_power_dbm=cfg.TX_POWER_DBM, freq_hz=cfg.FREQ_MHZ * 1e6,
    preamble_symbols=cfg.PREAMBLE_SYMS, explicit_header=cfg.EXPLICIT_HDR,
)
if not cfg.ENABLE_LDR_AUTO:
    p.low_dr_opt = False

receiver = UDPFrameReceiver(LISTEN_HOST, LISTEN_PORT, timeout=30.0)

print("[RX] Ожидание UDP пакетов...")

try:
    while True:  # Бесконечный цикл для GNS3
        payload = receiver.recv()
        
        if payload is None:
            print("[RX] Таймаут — пакет не получен")
            continue
        
        meta, tx_bits, rx_signal = unpack_frame(payload)
        n_bits = meta["n_bits"]
        snr = meta["snr_db"]

        print(f"[RX] Пакет получен, SNR={snr:.1f} дБ, {len(rx_signal)} отсчётов")

        rx_data, _ = phy_receive(rx_signal, p, n_bits // 8, enable_ToA=cfg.ENABLE_TOA)
        rx_bytes_padded = rx_data.ljust(n_bits // 8, b'\x00')
        rx_bits = np.unpackbits(np.frombuffer(rx_bytes_padded, dtype=np.uint8))[:n_bits]

        error_mask = tx_bits ^ rx_bits
        n_errors = int(np.sum(error_mask))
        ber = n_errors / n_bits

        print(f"[RX] Ошибок: {n_errors} из {n_bits} бит | BER = {ber:.4e}")

        if n_errors == 0:
            print("[RX] Статус: ПАКЕТ ПРИНЯТ БЕЗ ОШИБОК")
        else:
            print(f"[RX] Статус: ОШИБКИ ОБНАРУЖЕНЫ ({n_errors} бит повреждено)")

        if cfg.SHOW_BIT_MAP:
            print("\n[RX] Карта ошибок (по битам):")
            width = cfg.BIT_MAP_WIDTH
            for start in range(0, n_bits, width):
                chunk = error_mask[start:start + width]
                line = "".join('X' if e else '.' for e in chunk)
                print(f"  {start:5d}: {line}")
                
except KeyboardInterrupt:
    print("\n[RX] Остановка приемника")
finally:
    receiver.close()
    print("[RX] Завершено")