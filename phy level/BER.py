# ---------------------------------------------------------------------------
# 8. BER: теоретическая кривая
# ---------------------------------------------------------------------------
# Источник: Elshabrawy & Robert, IEEE Comm. Letters 2018
# "Closed-Form Approximation of LoRa Modulation BER Performance"

import math
import random
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

def BER_model(p: LoRa_Params):
    '''
    msgLen = 1000;
    EbN0dB = ...; % вектор значений ОСШ на бит (дБ); например от 0 до 15
    BER = zeros(..., ...); % вероятность битовой ошибки в результате моделирования для разных Eb/N0
    NumExp = 100*msgLen;

    for i = 1:1:length(EbN0dB)
        for  n = 1:NumExp
        % Передатчик (генерация случайных информационных битов, маппер, формирователь комплексной огибающей)
        % Канал с АБГШ (в зависмости от текущего EbN0dB(i)
        % Демодулятор
        % Расчет BER (сравнение принятых битов с переданными):
        BER(i) = BER(i) + sum(rec_bits~=bits)/msgLen;
        end
    end

    BER = BER / NumExp;
    '''
    NumIter = 10e4
    EbNodB = 40
    BER = [];
    rand = 0;
    for ebn0db in range(EbNodB):
        for signal in range(100):
            rand = random.randint(0,1)
            

    