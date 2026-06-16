import setup as cfg

from LoRa_ToA import *
from params import LoRaParams
from modulate import *
from LoRa_ToA import *
from demodulate import *
from LoRa_Chirp import *
from LoRa_Coding import *
from BER_teor import *
from channel import *
from phy import *

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

snr_range = np.arange(-30, -10, 2)   # нормальный диапазон SNR
BER_all   = []                       # список, а не 0
rng       = np.random.default_rng(cfg.RANDOM_SEED)  # один rng на всё

for SNR in snr_range:
    BER = []                         # сбрасываем для каждого SNR

    for N in range(1000):
        n_bits   = cfg.PAYLOAD_BYTES * 8
        tx_bits  = rng.integers(0, 2, n_bits, dtype=np.uint8)
        tx_bytes = np.packbits(tx_bits).tobytes()

        toa = compute_toa(len(tx_bytes), p)

        tx_signal, tx_meta = phy_transmit(
            tx_bytes, p,
            enable_ToA=cfg.ENABLE_TOA,
        )

        rx_signal, snr_actual = simulate_channel(
            tx_signal, p,
            distance_m      =cfg.DISTANCE_M,
            noise_figure_db =cfg.NOISE_FIGURE_DB,
            temperature_k   =cfg.TEMPERATURE_K,
            path_loss_exp   =cfg.PATH_LOSS_EXP,
            enable_awgn     =cfg.ENABLE_AWGN,
            enable_path_loss=cfg.ENABLE_PATH_LOSS,
            fixed_snr_db    = SNR,
        )

        rx_data, meta = phy_receive(
            rx_signal, p,
            len(tx_bytes),
            enable_ToA=cfg.ENABLE_TOA,
        )

        rx_bytes_padded = rx_data.ljust(len(tx_bytes), b'\x00')
        rx_bits = np.unpackbits(
            np.frombuffer(rx_bytes_padded, dtype=np.uint8)
        )[:n_bits]

        n_errors = int(np.sum(tx_bits ^ rx_bits))
        BER.append(n_errors / n_bits)  

    BER_all.append(np.mean(BER))        
    print(f"SNR = {SNR:+.1f} dB → BER = {BER_all[-1]:.4e}")

    # Тест без канала — BER должен быть строго 0
    tx_bits  = rng.integers(0, 2, n_bits, dtype=np.uint8)
    tx_bytes = np.packbits(tx_bits).tobytes()

    tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=cfg.ENABLE_TOA)

    # Приём без шума — подаём tx_signal напрямую
    rx_data, _ = phy_receive(tx_signal, p, len(tx_bytes), enable_ToA=cfg.ENABLE_TOA)

    rx_bytes_padded = rx_data.ljust(len(tx_bytes), b'\x00')
    rx_bits = np.unpackbits(np.frombuffer(rx_bytes_padded, dtype=np.uint8))[:n_bits]

    n_errors = int(np.sum(tx_bits ^ rx_bits))
    print(f"Без канала: ошибок {n_errors} из {n_bits} бит, BER = {n_errors/n_bits:.4f}")
    # Ожидается: 0 ошибок. Если не 0 — проблема в modulate/demodulate

BER_all = np.array(BER_all)
print(BER_all)


import matplotlib.pyplot as plt
from scipy.special import erfc

# --- Теоретическая кривая ---
# LoRa использует CSS-модуляцию, теоретически близкую к некогерентному BFSK
# Для сравнения строим BFSK (некогерентный) и BPSK как верхнюю/нижнюю границы

snr_linear = 10 ** (snr_range / 10)

# Некогерентный BFSK — ближайшая теория к LoRa CSS
ber_bfsk = 0.5 * np.exp(-snr_linear / 2)

# BPSK — теоретический минимум (лучший случай)
ber_bpsk = 0.5 * erfc(np.sqrt(snr_linear))

# --- График ---
plt.figure(figsize=(9, 6))

plt.semilogy(snr_range, BER_all, 'bo-', linewidth=2,
             markersize=6, label='LoRa симуляция')

plt.semilogy(snr_range, ber_bfsk, 'r--',
             linewidth=1.5, label='Некогерентный BFSK (теория)')

plt.semilogy(snr_range, ber_bpsk, 'g:',
             linewidth=1.5, label='BPSK (теория, лучший случай)')

# Порог приёма LoRa (обычно −7.5 дБ для SF7)
threshold_db = snr_threshold_db(p)
plt.axvline(x=threshold_db, color='orange', linestyle='-.', linewidth=1.2,
            label=f'Порог SNR ({threshold_db:.1f} дБ)')

plt.xlabel('SNR (дБ)', fontsize=12)
plt.ylabel('BER', fontsize=12)
plt.title(f'BER vs SNR — LoRa SF{cfg.SF} BW{cfg.BW_KHZ}kHz CR4/{4 + cfg.CR}', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.4)
plt.ylim([1e-5, 1])
plt.xlim([snr_range[0], snr_range[-1]])
plt.tight_layout()
plt.savefig('BER_result.png', dpi=150)
plt.show()

# Тест кан

tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=False)

for snr_test in [-10, 0, 10]:
    rx_signal, snr_actual = simulate_channel(
        tx_signal, p,
        distance_m      =cfg.DISTANCE_M,
        noise_figure_db =cfg.NOISE_FIGURE_DB,
        temperature_k   =cfg.TEMPERATURE_K,
        path_loss_exp   =cfg.PATH_LOSS_EXP,
        enable_awgn     =cfg.ENABLE_AWGN,
        enable_path_loss=cfg.ENABLE_PATH_LOSS,
        fixed_snr_db    =snr_test,
    )
    
    # Измеряем реальный SNR на выходе
    M        = 2 ** p.sf
    sig_pwr  = np.mean(np.abs(tx_signal)**2)
    noise    = rx_signal - tx_signal  # только шум
    noise_pwr = np.mean(np.abs(noise)**2)
    snr_measured = 10 * np.log10(sig_pwr / max(noise_pwr, 1e-12))
    
    print(f"fixed_snr_db={snr_test:+.0f} → snr_actual={snr_actual:.1f} → snr_measured={snr_measured:.1f}")