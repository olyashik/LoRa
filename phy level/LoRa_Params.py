# ---------------------------------------------------------------------------
# 1. Параметры передатчика
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class LoRaParams:
    ''' 
    Все настраиваемые параметры LoRa.
    Значения по умолчанию.
    '''
    sf: int   = 7          # Spreading Factor: 7..12
    bw: float = 500e3      # Bandwidth: 125 / 250 / 500 кГц
    cr: int   = 1          # Coding Rate denominator: 1→4/5, 2→4/6, 3→4/7, 4→4/8
    tx_power_dbm: float = 40.0   # мощность передатчика, дБм 
    freq_hz: float = 868.1e6     # несущая частота
    preamble_symbols: int = 8    # стандарт LoRaWAN: 8 символов преамбулы
    explicit_header: bool = True # явный заголовок (стандарт для LoRaWAN)
    low_dr_opt: bool = False      # Low Data Rate Optimize (включать при SF≥11, BW=125к)

    def __post_init__(self):
        assert 7 <= self.sf <= 12, "SF должен быть от 7 до 12"
        assert self.bw in (125e3, 250e3, 500e3), "BW: 125/250/500 кГц"
        assert 1 <= self.cr <= 4, "CR: 1..4"

        # Авто-включение LDR optimize согласно спецификации Semtech
        # Дополнительная функция (подходит для дешёвых модулей)
        if self.sf >= 11 and self.bw == 125e3:
            self.low_dr_opt = True
