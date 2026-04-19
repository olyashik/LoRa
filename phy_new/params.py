"""
lora_params.py — параметры конфигурации LoRa-трансивера
========================================================
Источник: Semtech SX1276 datasheet, Table 17 (RegModemConfig1/2/3)
"""

from dataclasses import dataclass



@dataclass
class LoRaParams:
    """
    Все настраиваемые параметры LoRa.
    Значения по умолчанию — типичная EU868 конфигурация.
    """
    sf: int             = 7        # Spreading Factor: 7..12
    bw: float           = 125e3    # Bandwidth: 125 / 250 / 500 кГц
    cr: int             = 1        # Coding Rate: 1→4/5, 2→4/6, 3→4/7, 4→4/8
    tx_power_dbm: float = 14.0     # мощность передатчика, дБм (макс EU: 14 дБм)
    freq_hz: float      = 868.1e6  # несущая частота
    preamble_symbols: int = 8      # стандарт LoRaWAN: 8 символов преамбулы
    explicit_header: bool = True   # явный заголовок (стандарт для LoRaWAN)
    low_dr_opt: bool    = False    # Low Data Rate Optimize (см. __post_init__)


    def __post_init__(self):
        assert 7 <= self.sf <= 12,               "SF должен быть от 7 до 12"
        assert self.bw in (125e3, 250e3, 500e3), "BW: 125/250/500 кГц"
        assert 1 <= self.cr <= 4,                "CR: 1..4"
        # Авто-включение LDR Optimize согласно спецификации Semtech:
        # при Ts = 2^SF / BW >= 16.38 мс дрейф кварца становится критичным.
        if self.sf >= 11 and self.bw == 125e3:
            self.low_dr_opt = True