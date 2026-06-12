import setup as cfg
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

from LoRa_ToA import *
from params import LoRaParams
from modulate import *
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

# ── Битовая скорость и сдвиг SNR → Eb/N0 ─────────────────────────────────
rb       = cfg.SF * (cfg.BW_KHZ * 1e3) / (2 ** cfg.SF) * (4 / (4 + cfg.CR))
shift_db = 10 * math.log10((cfg.BW_KHZ * 1e3) / rb)

print(f"SF={cfg.SF}, BW={cfg.BW_KHZ} кГц, CR=4/{4+cfg.CR}")
print(f"Rb = {rb:.1f} бит/с")
print(f"Сдвиг SNR → Eb/N0 = {shift_db:.1f} дБ")

# ── Диапазон SNR (подбираем так чтобы Eb/N0 был от -2 до +10 дБ) ──────────
eb_n0_min = -2
eb_n0_max = 10
snr_range = np.arange(eb_n0_min - shift_db, eb_n0_max - shift_db, 2)

print(f"SNR range:   {snr_range[0]:.1f} .. {snr_range[-1]:.1f} дБ")
print(f"Eb/N0 range: {snr_range[0]+shift_db:.1f} .. {snr_range[-1]+shift_db:.1f} дБ")

# ── Основной цикл BER ─────────────────────────────────────────────────────
BER_all = []
rng     = np.random.default_rng(cfg.RANDOM_SEED)

for SNR in snr_range:
    BER = []

    for N in range(1000):
        n_bits   = cfg.PAYLOAD_BYTES * 8
        tx_bits  = rng.integers(0, 2, n_bits, dtype=np.uint8)
        tx_bytes = np.packbits(tx_bits).tobytes()

        tx_signal, _ = phy_transmit(tx_bytes, p, enable_ToA=cfg.ENABLE_TOA)

        rx_signal, _ = simulate_channel(
            tx_signal, p,
            distance_m      =cfg.DISTANCE_M,
            noise_figure_db =cfg.NOISE_FIGURE_DB,
            temperature_k   =cfg.TEMPERATURE_K,
            path_loss_exp   =cfg.PATH_LOSS_EXP,
            enable_awgn     =cfg.ENABLE_AWGN,
            enable_path_loss=cfg.ENABLE_PATH_LOSS,
            fixed_snr_db    =SNR,
        )

        rx_data, _ = phy_receive(
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
    print(f"SNR={SNR:+.1f} дБ | Eb/N0={SNR+shift_db:+.1f} дБ | BER={BER_all[-1]:.4e}")

BER_all = np.array(BER_all)

# ── Теоретические кривые ──────────────────────────────────────────────────
eb_n0_range = snr_range + shift_db
snr_linear  = 10 ** (eb_n0_range / 10)

ber_bfsk = 0.5 * np.exp(-snr_linear / 2)
ber_bpsk = 0.5 * erfc(np.sqrt(snr_linear))

# ── График ────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 6))

plt.semilogy(eb_n0_range, BER_all, 'bo-', linewidth=2,
             markersize=6, label='LoRa симуляция')

plt.semilogy(eb_n0_range, ber_bfsk, 'r--',
             linewidth=1.5, label='Некогерентный BFSK (теория)')

plt.semilogy(eb_n0_range, ber_bpsk, 'g:',
             linewidth=1.5, label='BPSK (теория)')

threshold_db = snr_threshold_db(p)
plt.axvline(x=threshold_db + shift_db, color='orange', linestyle='-.', linewidth=1.2,
            label=f'Порог SNR ({threshold_db + shift_db:.1f} дБ)')

plt.xlabel('Eb/N0 (дБ)', fontsize=12)
plt.ylabel('BER', fontsize=12)
plt.title(
    f'BER vs Eb/N0 — LoRa SF{cfg.SF} BW{cfg.BW_KHZ}kHz CR4/{4+cfg.CR}\n'
    f'Rb={rb:.1f} бит/с',
    fontsize=12
)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.4)
plt.ylim([1e-5, 1])
plt.xlim([eb_n0_range[0], eb_n0_range[-1]])
plt.tight_layout()
plt.savefig('BER_result.png', dpi=150)
plt.show()


# Увеличить статистику 
# Найти учебник с теорией BER LoRa
# Включение выключение Pass Low 
# Прогнать всё, что правильно всё + виртуальная машина
