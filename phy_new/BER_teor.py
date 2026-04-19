
# ---------------------------------------------------------------------------
# 7. BER: теоретическая кривая
# ---------------------------------------------------------------------------
# Источник: Elshabrawy & Robert, IEEE Comm. Letters 2018

from params import LoRaParams
import math

def ber_theory(snr_db: float, p: LoRaParams) -> float:
    """
    Теоретический BER для LoRa (некогерентный приёмник) в канале AWGN.

    Приближение (Elshabrawy & Robert 2018):
        BER ≈ (2^SF / 2) / (2^SF - 1) × erfc(√(SF·SNR / 2))
    """
    from scipy.special import erfc
    snr_lin = 10 ** (snr_db / 10)
    M       = 2 ** p.sf
    ber     = ((M / 2) / (M - 1)) * erfc(math.sqrt(p.sf * snr_lin / 2))
    return min(max(ber, 1e-10), 0.5)


def snr_threshold_db(p: LoRaParams) -> float:
    """
    Пороговый SNR для приёма (из datasheet SX1276, Table 13).
    При этом SNR: BER ≈ 1% (стандарт Semtech).

    SF7:  −7.5 дБ   SF10: −15.0 дБ
    SF8:  −10.0 дБ  SF11: −17.5 дБ
    SF9:  −12.5 дБ  SF12: −20.0 дБ
    """
    return {7: -7.5, 8: -10.0, 9: -12.5,
            10: -15.0, 11: -17.5, 12: -20.0}[p.sf]
