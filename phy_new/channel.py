# ---------------------------------------------------------------------------
# Канал: путевые потери + AWGN + многолучевые замирания (Рэлей/Райс)
#        + гармоническая (тональная) узкополосная помеха
# ---------------------------------------------------------------------------

import numpy as np
from typing import Tuple, Optional
import math
from params import LoRaParams


# ---------------------------------------------------------------------------
# Путевые потери
# ---------------------------------------------------------------------------

def path_loss_db(distance_m: float, freq_hz: float,
                 path_loss_exp: float = 2.7) -> float:
    """
    Логарифмическая модель потерь.

    Фриис (свободное пространство, n=2):
        PL_0 = 20·log10(4π·f/c)   [дБ на 1 м]

    Общая модель:
        PL(d) = PL_0 + 10·n·log10(d)
    """
    c    = 3e8
    pl_0 = 20 * math.log10(4 * math.pi * freq_hz / c)
    return pl_0 + 10 * path_loss_exp * math.log10(max(distance_m, 1))


# ---------------------------------------------------------------------------
# AWGN
# ---------------------------------------------------------------------------

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Добавляет комплексный белый гауссовский шум (AWGN).

    SNR = P_signal / P_noise = 10^(snr_db/10)
    σ² = P_signal / SNR        (дисперсия шума)
    σ  = sqrt(P_signal / SNR)  (среднеквадратическое отклонение)

    Шум комплексный: n = (n_I + j·n_Q) / √2, оба компонента N(0,σ²/2)
    """
    p_signal = np.mean(np.abs(signal) ** 2)
    snr_lin  = 10 ** (snr_db / 10)
    sigma    = math.sqrt(p_signal / snr_lin)
    noise    = (np.random.randn(len(signal)) +
                1j * np.random.randn(len(signal))) * sigma / math.sqrt(2)
    return signal + noise


# ---------------------------------------------------------------------------
# Многолучевые замирания: Рэлей и Райс
#
# Общий генератор комплексного релеевского коэффициента затухания
# методом суммы синусоид Кларка/Джейкса в модификации Pop-Beaulieu [1,2,3].
#
#     X(t) = sqrt(2/N) * Σ_{n=1}^{N} cos(2π f_D t cos(α_n) + φ_n)
#     Y(t) = sqrt(2/N) * Σ_{n=1}^{N} sin(2π f_D t cos(α_n) + φ_n)
#     h(t) = (X(t) + j Y(t)) / √2                    ->  E[|h(t)|²] = 1
#
#     α_n = 2π n / N              (равномерно распределённые углы прихода)
#     φ_n ~ U(0, 2π)              (случайные начальные фазы, i.i.d.)
#
# При doppler_hz = f_D = 0 выражение вырождается в ОДИН случайный
# комплексный коэффициент на весь пакет (блочное / квазистатичное
# замирание) — это разумный дефолт для неподвижных узлов LoRa, так как
# время когерентности канала в такой ситуации намного больше длительности
# пакета, и предположение "один и тот же h на весь пакет" физически верно.
#
# При doppler_hz > 0 получается частотно-временное (time-selective)
# замирание — учитывает движение приёмника/передатчика или переотражающих
# объектов; f_D = v/λ, где v — относительная скорость, λ — длина волны.
# ---------------------------------------------------------------------------

def _clarke_jakes_gain(n_samples: int, fs: float, doppler_hz: float,
                        n_oscillators: int = 8,
                        rng: Optional[np.random.Generator] = None
                        ) -> np.ndarray:
    """
    Комплексный релеевский коэффициент замирания h(t), E[|h|^2] = 1.
    См. блок с формулами и источники [1], [2], [3] выше.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = n_oscillators
    t = np.arange(n_samples) / fs

    alpha_n = 2 * np.pi * np.arange(1, N + 1) / N        # углы прихода
    phi_n   = rng.uniform(0, 2 * np.pi, size=N)           # начальные фазы

    # форма (N, n_samples): по осцилляторам суммируем в axis=0
    phase = (2 * np.pi * doppler_hz * np.cos(alpha_n)[:, None] * t[None, :]
             + phi_n[:, None])

    X = np.sqrt(2.0 / N) * np.sum(np.cos(phase), axis=0)
    Y = np.sqrt(2.0 / N) * np.sum(np.sin(phase), axis=0)

    return (X + 1j * Y) / math.sqrt(2)


def add_rayleigh_fading(signal: np.ndarray, fs: float,
                         doppler_hz: float = 0.0,
                         n_oscillators: int = 8,
                         seed: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Релеевское замирание — нет доминирующего луча прямой видимости (NLOS),
    K = 0. Огибающая |h(t)| распределена по закону Рэлея:

        y(t) = h(t) · s(t)

    Параметры
    ---------
    fs           : частота дискретизации сигнала, Гц
    doppler_hz   : максимальная доплеровская частота f_D, Гц.
                   0 -> блочное (квазистатичное) замирание — один
                   случайный коэффициент на весь пакет.
    n_oscillators: число осцилляторов N в сумме синусоид (обычно 8-20
                   достаточно для хорошего приближения к теоретической
                   PSD Кларка, см. [1], [3]).
    seed         : сид ГСЧ для воспроизводимости.

    Возвращает
    ----------
    (y, h) — сигнал после замирания и сам комплексный коэффициент
    (пригодится для построения гистограммы |h| и сверки с теоретической
    PDF Рэлея: f(z) = (2z/σ²)·exp(-z²/σ²), σ²=1 при используемой
    нормировке).
    """
    rng = np.random.default_rng(seed)
    h = _clarke_jakes_gain(len(signal), fs, doppler_hz, n_oscillators, rng)
    return signal * h, h


def add_rician_fading(signal: np.ndarray, fs: float,
                       k_factor_db: float,
                       doppler_hz: float = 0.0,
                       los_freq_offset_hz: float = 0.0,
                       n_oscillators: int = 8,
                       seed: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Райсовское замирание — присутствует детерминированная составляющая
    прямой видимости (LOS) в дополнение к рэлеевскому рассеянию:

        h(t) = sqrt(K/(K+1)) · e^{j(2π f_LOS t + φ0)}
             + sqrt(1/(K+1)) · h_Rayleigh(t)

    K = P_LOS / P_scatter — фактор Райса (в линейном масштабе):
        K = 0     -> вырождается в чистый Рэлей
        K → ∞     -> канал без замираний (чистый LOS)
    Типичные значения для частично затенённой LOS-линии: 3..15 дБ.

    См. источники [4], [5], [6].

    Параметры
    ---------
    k_factor_db        : фактор Райса K, дБ.
    doppler_hz          : доплеровский разброс рассеянной составляющей, Гц.
    los_freq_offset_hz  : доплеровский сдвиг ЛУЧА ПРЯМОЙ ВИДИМОСТИ, Гц
                          (например, из-за движения приёмника вдоль линии
                          визирования; для статичного узла = 0).

    Возвращает
    ----------
    (y, h) — сигнал после замирания и сам коэффициент.
    """
    rng = np.random.default_rng(seed)
    k_lin = 10 ** (k_factor_db / 10)

    h_rayleigh = _clarke_jakes_gain(len(signal), fs, doppler_hz,
                                     n_oscillators, rng)

    t    = np.arange(len(signal)) / fs
    phi0 = rng.uniform(0, 2 * np.pi)
    h_los = np.exp(1j * (2 * np.pi * los_freq_offset_hz * t + phi0))

    h = (math.sqrt(k_lin / (k_lin + 1)) * h_los +
         math.sqrt(1 / (k_lin + 1)) * h_rayleigh)

    return signal * h, h


# ---------------------------------------------------------------------------
# Гармоническая (тональная / многочастотная) помеха
#
#     i(t) = Σ_{k=1}^{M} A_k · cos(2π k·f0·t + φ_k),   A_k = A_1 · decay^{k-1}
#
# Классическая модель узкополосной детерминированной помехи внутри полосы
# приёма — тактовые генераторы, ШИМ-преобразователи, гетеродины соседних
# передатчиков и т.п. При M=1 — чистый тон (CW-помеха), при M>1 — гребёнка
# гармоник с затуханием decay по номеру гармоники.
#
# Уровень задаётся через SIR (signal-to-interference ratio) — по аналогии
# с тем, как SNR задаётся в add_awgn():
#
#     SIR = P_signal / P_interf  [линейно],   SIR_дБ = 10·log10(SIR)
#
# Источники: [7], [8] — конкретно для LoRa/CSS; [9], [10] — общая теория.
# ---------------------------------------------------------------------------

def add_harmonic_interference(signal: np.ndarray, fs: float,
                               sir_db: float,
                               freq_offset_hz: float,
                               n_harmonics: int = 1,
                               harmonic_decay: float = 0.5,
                               phase_rad: Optional[float] = None,
                               seed: Optional[int] = None) -> np.ndarray:
    """
    Добавляет гармоническую (тональную/многочастотную) помеху.
    См. блок с формулами и источники [7]-[10] выше.

    Параметры
    ---------
    fs              : частота дискретизации, Гц.
    sir_db          : требуемое соотношение сигнал/помеха, дБ.
    freq_offset_hz  : смещение ОСНОВНОГО тона (f0) относительно центра
                      канала, Гц. Может лежать как внутри полосы Bw
                      (максимальное влияние на демодуляцию), так и за
                      её пределами (тогда реальный приёмник ослабил бы
                      помеху фильтром — в этой модели такое ослабление
                      не учитывается, считаем "помеха уже прошла ПФ").
    n_harmonics     : число гармоник M (1 = чистый тон).
    harmonic_decay  : коэффициент затухания амплитуды по номеру
                      гармоники (A_k = A_1 · decay^(k-1)).
    phase_rad       : фиксированная начальная фаза для всех гармоник;
                      если None — случайная равномерная фаза для каждой.
    seed            : сид ГСЧ.

    Возвращает
    ----------
    signal + i(t) — сигнал со сложенной помехой.
    """
    rng = np.random.default_rng(seed)
    n = len(signal)
    t = np.arange(n) / fs

    p_signal = np.mean(np.abs(signal) ** 2)
    sir_lin  = 10 ** (sir_db / 10)
    p_interf = p_signal / sir_lin

    # суммарная мощность гребёнки гармоник (комплексные экспоненты,
    # мощность каждой A_k^2): Σ A_k² = A_1² · Σ decay^(2(k-1)) = p_interf
    decay_sq_sum = sum(harmonic_decay ** (2 * (k - 1))
                       for k in range(1, n_harmonics + 1))
    a1 = math.sqrt(p_interf / decay_sq_sum)

    if phase_rad is None:
        phases = rng.uniform(0, 2 * np.pi, size=n_harmonics)
    else:
        phases = np.full(n_harmonics, phase_rad)

    interf = np.zeros(n, dtype=complex)
    for k in range(1, n_harmonics + 1):
        a_k = a1 * (harmonic_decay ** (k - 1))
        interf += a_k * np.exp(1j * (2 * np.pi * k * freq_offset_hz * t
                                      + phases[k - 1]))

    return signal + interf


# ---------------------------------------------------------------------------
# Сборка канала целиком
# ---------------------------------------------------------------------------

def simulate_channel(signal: np.ndarray, p: LoRaParams,
                     distance_m: float,
                     noise_figure_db: float = 6.0,
                     temperature_k: float = 290.0,
                     path_loss_exp: float = 2.7,
                     enable_awgn: bool = True,
                     enable_path_loss: bool = True,
                     fixed_snr_db: float = 10.0,
                     # --- многолучевые замирания ---
                     enable_rayleigh: bool = False,
                     enable_rician: bool = False,
                     k_factor_db: float = 6.0,
                     doppler_hz: float = 0.0,
                     los_freq_offset_hz: float = 0.0,
                     n_oscillators: int = 8,
                     # --- гармоническая помеха ---
                     enable_harmonic: bool = False,
                     harmonic_sir_db: float = 10.0,
                     harmonic_freq_offset_hz: float = 5000.0,
                     harmonic_n: int = 1,
                     harmonic_decay: float = 0.5,
                     # --- служебное ---
                     fading_seed: Optional[int] = None
                     ) -> Tuple[np.ndarray, float]:
    """
    Полная модель канала. Порядок применения эффектов физически важен:

      1. Многолучевое замирание (Рэлей/Райс) — МУЛЬТИПЛИКАТИВНОЕ
         воздействие на сам полезный сигнал (h(t)·s(t)), происходит
         "в эфире" по пути распространения.
      2. Гармоническая помеха и AWGN — АДДИТИВНЫЕ воздействия на входе
         приёмника (i(t) и n(t) складываются с уже замирающим сигналом).

    ВАЖНО: fs (частота дискретизации) берётся из p.fs. Если в твоём
    params.py атрибут называется иначе (например p.sample_rate) —
    поправь строку ниже.
    """
    fs = getattr(p, "fs", None)
    if fs is None:
        fs = p.bw  # запасной вариант — ЗАМЕНИ, если есть отдельный атрибут sample rate

    out = signal.copy()

    # 1. Многолучевое замирание
    if enable_rician:
        out, _ = add_rician_fading(out, fs, k_factor_db, doppler_hz,
                                    los_freq_offset_hz, n_oscillators,
                                    seed=fading_seed)
    elif enable_rayleigh:
        out, _ = add_rayleigh_fading(out, fs, doppler_hz, n_oscillators,
                                      seed=fading_seed)

    if not enable_awgn:
        # Идеальный по шуму канал — но помеха всё ещё может быть включена
        if enable_harmonic:
            out = add_harmonic_interference(out, fs, harmonic_sir_db,
                                             harmonic_freq_offset_hz,
                                             harmonic_n, harmonic_decay,
                                             seed=fading_seed)
        return out, fixed_snr_db

    if enable_path_loss:
        # Физическая модель: SNR считается по расстоянию
        k_b           = 1.38e-23
        n_thermal_dbm = 10 * math.log10(k_b * temperature_k * p.bw) + 30
        pl            = path_loss_db(distance_m, p.freq_hz, path_loss_exp)
        p_rx_dbm      = p.tx_power_dbm - pl

        snr_db        = p_rx_dbm - n_thermal_dbm - noise_figure_db
    else:
        # Используем fixed_snr_db напрямую — режим для BER-симуляции
        snr_db = fixed_snr_db

    # 2. Аддитивные воздействия на входе приёмника
    if enable_harmonic:
        out = add_harmonic_interference(out, fs, harmonic_sir_db,
                                         harmonic_freq_offset_hz,
                                         harmonic_n, harmonic_decay,
                                         seed=fading_seed)

    out = add_awgn(out, snr_db)

    return out, snr_db

# комплексное сопряжение в питоне (функции)
# почитать учебник по матлабу
# место, где потерялся канал
# на физ.уровне мы всё равно работаем с цифрой