# ---------------------------------------------------------------------------
# 1. Time on Air (ToA)
# ---------------------------------------------------------------------------
# Источник: Semtech AN1200.22, раздел "LoRa Modem Time-on-Air"
# и SX1276 datasheet, формулы на стр. 31-32.

import math
from params import LoRaParams

def compute_toa(payload_bytes: int, p: LoRaParams) -> dict:
    """
    Вычисляет Time on Air по формулам Semtech.

    Формула символьной скорости:
        Rs = BW / 2^SF   [символов/сек]

    Длительность одного символа:
        Ts = 1 / Rs = 2^SF / BW   [сек]

    Число символов преамбулы (с дробной частью 4.25):
        N_preamble = (preamble_symbols + 4.25) × Ts

    Число символов полезной нагрузки:
        N_payload = 8 + max(
            ceil( (8×PL - 4×SF + 28 + 16×CRC - 20×IH) /
                  (4×(SF - 2×LDR)) ) × (CR + 4),
            0 )
    где:
        PL  = число байт полезной нагрузки
        CRC = 1 (CRC включён, стандарт)
        IH  = 0 если явный заголовок (explicit header)
        LDR = 1 если Low Data Rate Optimize включён
    """
    Ts = (2 ** p.sf) / p.bw

    # Число символов полезной нагрузки
    crc  = 1                          # CRC всегда включён в LoRaWAN
    ih   = 0 if p.explicit_header else 1
    ldr  = 1 if p.low_dr_opt else 0

    numerator   = 8 * payload_bytes - 4 * p.sf + 28 + 16 * crc - 20 * ih
    denominator = 4 * (p.sf - 2 * ldr)
    payload_sym = 8 + max(math.ceil(numerator / denominator) * (p.cr + 4), 0)

    # Итоговый ToA
    t_preamble = (p.preamble_symbols + 4.25) * Ts
    t_payload  = payload_sym * Ts
    toa        = t_preamble + t_payload

    return {
        "toa_s":              round(toa, 6),
        "toa_ms":             round(toa * 1000, 3),
        "t_preamble_ms":      round(t_preamble * 1000, 3),
        "t_payload_ms":       round(t_payload * 1000, 3),
        "payload_symbols":    payload_sym,
        "symbol_duration_ms": round(Ts * 1000, 4),
        "symbol_rate_bps":    round(p.sf * p.bw / (2 ** p.sf), 2),
    }
