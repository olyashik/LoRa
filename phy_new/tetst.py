from channel import simulate_channel, add_awgn
from phy import phy_transmit, phy_receive
from modulate import modulate
from demodulate import demodulate, demodulate_symbol
from LoRa_Chirp import generate_chirp
import numpy as np
from params import LoRaParams
from setup import *

import setup as cfg
from params import LoRaParams

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

rng = np.random.default_rng(42)
M   = 2 ** p.sf

# ── ТЕСТ A: один символ через чирп + шум ──────────────────────────
print("\n[A] Символ → чирп → шум → демодуляция:")
for snr_test in [20, 10, 0, -10]:
    errors = 0
    N_test = 500
    for _ in range(N_test):
        sym      = int(rng.integers(0, M))
        chirp    = generate_chirp(sym, p)
        noisy    = add_awgn(chirp, snr_test)
        sym_rx   = demodulate_symbol(noisy, p)
        if sym_rx != sym:
            errors += 1
    print(f"  SNR={snr_test:+3d} дБ → SER={errors/N_test:.4f}  ({errors}/{N_test} ошибок)")

# ── ТЕСТ B: полный пайплайн с шумом ───────────────────────────────
print("\n[B] Полный пайплайн с шумом (transmit → channel → receive):")
for snr_test in [20, 10, 0, -10]:
    errors = 0
    n_bits = PAYLOAD_BYTES * 8
    N_test = 200
    for _ in range(N_test):
        tx_bits  = rng.integers(0, 2, n_bits, dtype=np.uint8)
        tx_bytes = np.packbits(tx_bits).tobytes()

        tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=False)
        rx_signal    = add_awgn(tx_signal, snr_test)  # напрямую, без simulate_channel
        rx_data, _   = phy_receive(rx_signal, p, len(tx_bytes), enable_ToA=False)

        rx_bytes_padded = rx_data.ljust(len(tx_bytes), b'\x00')
        rx_bits = np.unpackbits(
            np.frombuffer(rx_bytes_padded, dtype=np.uint8)
        )[:n_bits]

        errors += int(np.sum(tx_bits ^ rx_bits))

    ber = errors / (N_test * n_bits)
    print(f"  SNR={snr_test:+3d} дБ → BER={ber:.4f}")

print("\n[C] Проверка offset в phy_receive:")
tx_bytes     = bytes(rng.integers(0, 256, cfg.PAYLOAD_BYTES, dtype=np.uint8))
tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=False)

# Без шума
rx_data, meta = phy_receive(tx_signal, p, len(tx_bytes), enable_ToA=False)
print(f"  Без шума:  sync_offset={meta.get('sync_offset')}, symbols={meta.get('symbols_decoded')}, ok={rx_data == tx_bytes}")

# С шумом
for snr_test in [20, 0, -10]:
    noisy         = add_awgn(tx_signal, snr_test)
    rx_data, meta = phy_receive(noisy, p, len(tx_bytes), enable_ToA=False)
    print(f"  SNR={snr_test:+3d}: sync_offset={meta.get('sync_offset')}, symbols={meta.get('symbols_decoded')}, ok={rx_data == tx_bytes}")

# Принудительно с правильным offset
print("\n[D] phy_receive с правильным offset (минуя синхронизацию):")
M      = 2 ** p.sf
offset = (p.preamble_symbols + 2) * M   # точный offset

for snr_test in [20, 0, -10]:
    noisy          = add_awgn(tx_signal, snr_test)
    payload_signal = noisy[offset:]
    n_symbols      = len(payload_signal) // M

    from demodulate import demodulate_symbol, demodulate
    import numpy as np
    symbols = np.array([
        demodulate_symbol(payload_signal[i*M:(i+1)*M], p)
        for i in range(n_symbols)
    ], dtype=np.int32)
    rx_data = demodulate(symbols, p)[:len(tx_bytes)]
    
    err = sum(a != b for a, b in zip(tx_bytes, rx_data))
    print(f"  SNR={snr_test:+3d}: ошибок байт {err} из {len(tx_bytes)}")
