# ---------------------------------------------------------------------------
# 9. Главный интерфейс PHY (используется MAC-уровнем)
# ---------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional
from Demodulation import *

def phy_transmit(data: bytes, p: LoRa_Params,
                 fs: float = None) -> Tuple[np.ndarray, dict]:
    """
    Точка входа MAC → PHY (передача).

    MAC передаёт сырые байты (PHYPayload из LoRaWAN-фрейма).
    PHY возвращает I/Q сигнал и метаданные.

    Возвращает:
        signal  — комплексный сигнал np.ndarray
        meta    — словарь с ToA, символьной скоростью и т.д.
    """
    signal = modulate(data, p, fs)
    meta   = compute_toa(len(data), p)
    meta["n_payload_bytes"] = len(data)
    meta["sf"] = p.sf
    meta["bw_khz"] = p.bw / 1e3
    meta["cr"] = f"4/{4 + p.cr}"
    meta["tx_power_dbm"] = p.tx_power_dbm
    return signal, meta


def phy_receive(rx_signal: np.ndarray, p: LoRa_Params,
                n_payload_bytes: int,
                fs: float = None) -> Tuple[Optional[bytes], dict]:
    """
    Точка входа PHY → MAC (приём).

    PHY принимает I/Q сигнал и возвращает байты + метаданные.
    MAC использует meta["snr_est_db"] для ADR.

    Возвращает:
        data  — байты или None при ошибке декодирования
        meta  — snr_est_db, rssi_est_dbm, toa, symbols_decoded
    """
    return demodulate(rx_signal, p, n_payload_bytes, fs)