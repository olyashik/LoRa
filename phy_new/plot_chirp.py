# ---------------------------------------------------------------------------
# plot_chirp.py — визуализация чирп-символов LoRa (не требует симуляции
# канала, работает быстро).
#
# Использует твой generate_chirp()/generate_base_chirp() из LoRa_Chirp.py,
# fs = p.bw (та же дискретизация, что и в реальном модуляторе/демодуляторе,
# N = 2^SF отсчётов на символ) — график честно отражает то, что реально
# идёт по пайплайну.
#
# Источники:
#   Vangelista, L. "Frequency Shift Chirp Modulation: The LoRa Modulation",
#   IEEE Signal Process. Lett., 24(12), 2017, pp. 1818-1821.
#   Semtech AN1200.22 "LoRa Modulation Basics", 2015.
#   Патент Semtech EP2763321 (аналитическая модель чирпа).
#
# Запуск: python plot_chirp.py
# ---------------------------------------------------------------------------

import setup as cfg
import numpy as np
import matplotlib.pyplot as plt

from params import LoRaParams
from LoRa_Chirp import generate_chirp, generate_base_chirp


def _inst_freq(sig, fs):
    """Мгновенная частота = производная развёрнутой фазы / (2π)."""
    phase = np.unwrap(np.angle(sig))
    return np.diff(phase) / (2 * np.pi) * fs


def plot_lora_chirp(p, symbol_value=None, save_path="LoRa_chirp.png"):
    """
    Строит:
      1) мгновенную частоту up-chirp(symbol=0) и up-chirp(symbol=symbol_value)
         -> виден "wrap-around" (+BW/2 -> -BW/2), кодирующий значение символа
      2) I/Q компоненты базового up-chirp (symbol=0)
      3) спектрограмму последовательности up(0) -> up(symbol_value) -> down(0)
    """
    M = 2 ** p.sf
    T = M / p.bw
    t = np.arange(M) / p.bw

    if symbol_value is None:
        symbol_value = M // 3   # ненулевой символ для демонстрации wrap-around

    up0   = generate_base_chirp(p)                    # symbol=0
    upS   = generate_chirp(symbol_value, p, up=True)   # произвольный символ
    down0 = generate_chirp(0, p, up=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 1. Мгновенная частота: symbol=0 vs symbol=symbol_value
    axes[0].plot(t[:-1] * 1e3, _inst_freq(up0, p.bw) / 1e3,
                 label='up-chirp, symbol=0', color='tab:blue')
    axes[0].plot(t[:-1] * 1e3, _inst_freq(upS, p.bw) / 1e3,
                 label=f'up-chirp, symbol={symbol_value}', color='tab:green')
    axes[0].set_xlabel('t, мс'); axes[0].set_ylabel('f, кГц')
    axes[0].set_title(
        f'Мгновенная частота (SF={p.sf}, BW={p.bw/1e3:.0f} кГц, '
        f'T_sym={T*1e3:.3f} мс)\n'
        f'ненулевой символ смещает начальную частоту и даёт скачок '
        f'(+BW/2 → −BW/2) в момент "переполнения"'
    )
    axes[0].legend(); axes[0].grid(alpha=0.4)

    # 2. I/Q базового up-chirp
    axes[1].plot(t * 1e3, np.real(up0), linewidth=0.8, label='I = Re{s(t)}')
    axes[1].plot(t * 1e3, np.imag(up0), linewidth=0.8, label='Q = Im{s(t)}', alpha=0.7)
    axes[1].set_xlabel('t, мс'); axes[1].set_ylabel('амплитуда')
    axes[1].set_title('Up-chirp (symbol=0) — компоненты I/Q, |s[n]|=1 (const. envelope)')
    axes[1].legend(); axes[1].grid(alpha=0.4)

    # 3. Спектрограмма последовательности символов
    seq = np.concatenate([up0, upS, down0])
    nfft = 64 if M >= 256 else 16   # для маленьких SF (короткий символ) окно меньше
    axes[2].specgram(seq, NFFT=nfft, Fs=p.bw, noverlap=nfft - 16, cmap='viridis')
    axes[2].set_xlabel('t, с'); axes[2].set_ylabel('f, Гц')
    axes[2].set_title(
        f'Спектрограмма: up(0) → up({symbol_value}) → down(0)\n'
        f'(развёртка данных + характерный "V" от смены знака в down-chirp)'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main(sf=9):
    p = LoRaParams(
        sf               = sf,
        bw               = cfg.BW_KHZ * 1e3,
        cr               = cfg.CR,
        tx_power_dbm     = cfg.TX_POWER_DBM,
        freq_hz          = cfg.FREQ_MHZ * 1e6,
        preamble_symbols = cfg.PREAMBLE_SYMS,
        explicit_header  = cfg.EXPLICIT_HDR,
    )
    plot_lora_chirp(p)


if __name__ == "__main__":
    main(sf=9)