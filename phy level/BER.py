# ber_simulation.py
import numpy as np
from typing import Tuple

from PHY import (
    bits_to_symbols,
    symbols_to_bits,
    modulate,
    demodulate,
    add_awgn
)


def simulate_ber(p, msg_len: int,
                 EbN0dB: np.ndarray,
                 NumExp: int = 1000,
                 gray: bool = True) -> np.ndarray:
    """
    Монте-Карло BER симуляция.

    BER(i) = среднее по NumExp экспериментов:
        sum(errors/msg_len)
    """

    sf = p.sf
    BER = np.zeros(len(EbN0dB))

    rng = np.random.default_rng()

    for i, snr_db in enumerate(EbN0dB):

        for _ in range(NumExp):

            # --- TX ---
            bits = rng.integers(0, 2, msg_len, dtype=np.uint8)

            n_sym = len(bits) // sf
            bits = bits[:n_sym * sf]

            syms = bits_to_symbols(bits, sf, gray=gray)
            tx_signal = modulate(syms, p)

            # --- CHANNEL ---
            rx_signal = add_awgn(tx_signal, snr_db)

            # --- RX ---
            rx_syms = demodulate(rx_signal, p, len(syms))
            rx_bits = symbols_to_bits(rx_syms, sf, gray=gray)

            # --- BER ---
            BER[i] += np.sum(rx_bits != bits) / msg_len

        BER[i] /= NumExp

    return BER