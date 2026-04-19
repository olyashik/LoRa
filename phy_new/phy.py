# ---------------------------------------------------------------------------
# 8. Главный интерфейс PHY (используется MAC-уровнем)
# ---------------------------------------------------------------------------

from typing import Tuple, Optional
from params import LoRaParams
from modulate import *
from LoRa_ToA import *
from demodulate import *

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