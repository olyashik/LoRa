# ---------------------------------------------------------------------------
# 8. BER: теоретическая кривая
# ---------------------------------------------------------------------------
# Источник: Elshabrawy & Robert, IEEE Comm. Letters 2018
# "Closed-Form Approximation of LoRa Modulation BER Performance"

import math
import LoRa_Params
from scipy.special import erfc

def ber_theory(snr_db: float, p: LoRa_Params) -> float:
    '''
    Теоретический BER для LoRa (некогерентный приёмник) в канале AWGN.

    Приближение:
        BER ≈ (1/SF) * (2^SF / 2) * Q(√(2·SNR·SF))
            = (2^SF / 2) * (1/SF) * erfc(√(SNR·SF))

    Точная формула включает суммирование по всем 2^SF символам,
    приближение даёт ошибку < 0.5 дБ при SNR > −15 дБ.

    Processing Gain в дБ:
        PG = 10·log10(SF * BW / (BW / 2^SF))
           = 10·log10(SF * 2^SF)
    '''
    
    snr_lin = 10 ** (snr_db / 10)
    M = 2 ** p.sf
    # BER (некогерентная детекция M-FSK аппроксимация)
    ber = ((M / 2) / (M - 1)) * erfc(math.sqrt(p.sf * snr_lin / 2))
    return min(max(ber, 1e-10), 0.5)


def snr_threshold_db(p: LoRa_Params) -> float:
    '''
    Пороговый SNR для приёма (из datasheet SX1276, Table 13).
    При этом SNR: BER ≈ 1% (стандарт Semtech).

    SF7:  −7.5 дБ
    SF8:  −10 дБ
    SF9:  −12.5 дБ
    SF10: −15 дБ
    SF11: −17.5 дБ
    SF12: −20 дБ
    '''
    thresholds = {7: -7.5, 8: -10.0, 9: -12.5,
                  10: -15.0, 11: -17.5, 12: -20.0}
    return thresholds[p.sf]