# ---------------------------------------------------------------------------
# 6. Time on Air
# ---------------------------------------------------------------------------
# Источник: Semtech AN1200.22; SX1276 datasheet стр. 31-32

from LoRa_Params import LoRaParams
import math

def compute_toa(payload_bytes: int, p: LoRaParams) -> dict:
    """
    Time on Air по формулам Semtech.

        Ts = 2^SF / BW

        N_payload = 8 + max(
            ⌈(8·PL − 4·SF + 28 + 16·CRC − 20·IH) / (4·(SF − 2·LDR))⌉ · (CR+4),
            0)

        ToA = (N_pre + 4.25)·Ts + N_payload·Ts
    """
    Ts  = (2 ** p.sf) / p.bw
    ih  = 0 if p.explicit_header else 1
    ldr = 1 if p.low_dr_opt else 0
    num = 8 * payload_bytes - 4 * p.sf + 28 + 16 - 20 * ih    # CRC=1 всегда
    den = 4 * (p.sf - 2 * ldr)
    n_pay = 8 + max(math.ceil(num / den) * (p.cr + 4), 0)
    t_pre = (p.preamble_symbols + 4.25) * Ts
    t_pay = n_pay * Ts
    return {
        "toa_ms":             round((t_pre + t_pay) * 1000, 3),
        "t_preamble_ms":      round(t_pre * 1000, 3),
        "t_payload_ms":       round(t_pay * 1000, 3),
        "payload_symbols":    n_pay,
        "symbol_duration_ms": round(Ts * 1000, 4),
        "symbol_rate_bps":    round(p.sf * p.bw / (2 ** p.sf), 2),
    }

