# -*- coding: utf-8 -*-
"""
data_generator.py
=================
Uretilenler:
  1) GERCEK arac veritabani (Net kWh / Max DC kW) - hard-coded gercek spektler.
  2) KALICI arac filosu (sabit ID'ler) - 100 gun boyunca ayni araclarin
     defalarca sarj olmasi (kumulatif SOH analizi icin sart).
  3) Stokastik sarj oturumlari (vehicle_id, gun, gelis SoC, gelis/kalkis).
  4) Tesis BAZ YUK profili - tepe noktasi trafonun ~%70'i (≈1120 kW).
  5) EPIAS gerceklerine yakin PTF (Piyasa Takas Fiyati) ve SMF (Sistem Marjinal
     Fiyati) egrileri (TL/MWh), saatlik -> dakikaya genisletilmis.

Tum stokastik buyuklukler NumPy ile vektorize uretilir.
Veriler data/ klasorune Parquet (yoksa CSV) olarak kaydedilir.
"""

from __future__ import annotations

import os
import datetime as _dt
from typing import Tuple

import numpy as np
import pandas as pd

from .config import MINUTES_PER_DAY, SimConfig, ScenarioConfig


# --------------------------------------------------------------------------- #
# 1) GERCEK ARAC VERITABANI  (Net kWh , Max DC kW)
# --------------------------------------------------------------------------- #
VEHICLE_DB = pd.DataFrame(
    [
        # --- Binek (AVM agirlikli) ---
        ("Togg T10X Long Range",      88.5, 150.0, "passenger"),
        ("Tesla Model Y Long Range",  75.0, 250.0, "passenger"),
        ("Tesla Model 3 LR",          75.0, 250.0, "passenger"),
        ("Renault Zoe",               52.0,  50.0, "passenger"),
        ("Porsche Taycan Plus",       93.4, 270.0, "passenger"),
        ("Hyundai Ioniq 5",           77.4, 233.0, "passenger"),
        ("Kia EV6 GT-Line",           77.4, 240.0, "passenger"),
        ("Volkswagen ID.4 Pro",       77.0, 135.0, "passenger"),
        ("BMW i4 eDrive40",           80.7, 205.0, "passenger"),
        ("Mercedes EQS 450+",        107.8, 200.0, "passenger"),
        ("Audi e-tron GT",            93.4, 270.0, "passenger"),
        ("Renault Megane E-Tech",     60.0, 130.0, "passenger"),
        ("MG4 Electric 64",           61.7, 135.0, "passenger"),
        ("BYD Atto 3",                60.5,  88.0, "passenger"),
        ("Volvo XC40 Recharge",       78.0, 150.0, "passenger"),
        ("Skoda Enyaq iV 80",         77.0, 135.0, "passenger"),
        # --- Ticari / filo (FABRIKA agirlikli) ---
        ("Ford E-Transit",            68.0, 115.0, "fleet"),
        ("Mercedes eSprinter",       113.0, 115.0, "fleet"),
        ("Fiat E-Ducato",             79.0,  50.0, "fleet"),
        ("Peugeot e-Expert",          75.0, 100.0, "fleet"),
        ("Maxus eDeliver 9",          88.5,  90.0, "fleet"),
        ("Renault Master E-Tech",     87.0,  46.0, "fleet"),
    ],
    columns=["model", "capacity", "dc_max", "segment"],
)


# --------------------------------------------------------------------------- #
# 2) KALICI ARAC FILOSU (sabit ID'ler)
# --------------------------------------------------------------------------- #
def build_fleet(cfg: SimConfig) -> pd.DataFrame:
    """
    Senaryoya gore sabit kimlikli arac filosu uretir.
    Her araç 100 gun boyunca tekrar tekrar bu kimlikle sarj olur.
    """
    rng = np.random.default_rng(cfg.seed + 101)
    sc = cfg.scenario
    n = sc.fleet_size()
    if sc.is_factory:
        weights = np.where(VEHICLE_DB["segment"].values == "fleet", 5.0, 0.4)
    else:
        weights = np.where(VEHICLE_DB["segment"].values == "passenger", 5.0, 0.3)
    weights = weights / weights.sum()
    idx = rng.choice(len(VEHICLE_DB), size=n, p=weights)
    chosen = VEHICLE_DB.iloc[idx].reset_index(drop=True)
    fleet = pd.DataFrame({
        "vehicle_id": [f"{'FLT' if sc.is_factory else 'EV'}-{i:03d}" for i in range(n)],
        "model": chosen["model"].values,
        "segment": chosen["segment"].values,
        "capacity": chosen["capacity"].values.astype(np.float64),
        "dc_max": chosen["dc_max"].values.astype(np.float64),
    })
    return fleet


# --------------------------------------------------------------------------- #
# Ortak GUNLUK TALEP/MEVSIM faktoru (A7: PTF-talep korelasyonu + mevsimsellik)
# --------------------------------------------------------------------------- #
def daily_demand_factor(cfg: SimConfig) -> np.ndarray:
    """
    Gunluk ortak talep/mevsim carpani (days,). HEM baz yuk HEM PTF bu sinyalle
    olceklenir; boylece yuksek-talep gunlerinde baz yuk de PTF de yukselir
    (gercek dunyadaki TALEP-FIYAT KORELASYONU). Yavas sinus = mevsimsel drift
    (sim penceresi boyunca ~1 dongu), uzerine gunluk hava gurultusu.
    Deterministiktir (cfg.seed); iki uretici ayni diziyi kullanir.
    """
    days = cfg.days
    d = np.arange(days)
    season = 0.12 * np.sin(2 * np.pi * d / max(days, 1))     # mevsimsel yavas drift
    rng = np.random.default_rng(cfg.seed + 333)
    weather = rng.normal(0.0, 0.07, size=days)               # gunluk hava/talep sapmasi
    return np.clip(1.0 + season + weather, 0.70, 1.35)


# --------------------------------------------------------------------------- #
# 2b) GERCEK ANKARA ORTAM SICAKLIGI (madde 1) - termal model girdisi
# --------------------------------------------------------------------------- #
def _sim_start_date(cfg: SimConfig) -> _dt.date:
    """Simulasyon baslangic tarihi (EN KOTU senaryo: Mayis basi)."""
    th = cfg.thermal
    # Yil onemsiz (sadece takvim ay/gun kullanilir); arti-yil sarmalanir.
    return _dt.date(2025, int(th.sim_start_month), int(th.sim_start_day))


def ambient_series(cfg: SimConfig) -> np.ndarray:
    """
    Simulasyon penceresi icin dakika bazli GERCEK Ankara ortam sicakligi θa(t) (°C).

    - Baslangic: Mayis basi (EN KOTU senaryo; 100 gun -> ~9 Agustos, en sicak bant).
    - Her gun, takvim ayinin Ankara ORTALAMA gunluk sicakligi alinir; aylar arasi
      yumusak gecis icin komsu ayla lineer harmanlama yapilir.
    - Gun-ici: θa = gun_ort + ay_genligi · sin(2π(saat−9)/24)  -> tepe ~15:00.
    - Uzerine kucuk, DETERMINISTIK gunluk hava sapmasi (cfg.seed).
    Donus: shape (days*1440,)
    """
    th = cfg.thermal
    days = cfg.days
    T = days * MINUTES_PER_DAY
    start = _sim_start_date(cfg)
    means = np.asarray(th.ankara_monthly_mean_c, dtype=np.float64)
    amps = np.asarray(th.ankara_monthly_amp_c, dtype=np.float64)

    rng = np.random.default_rng(cfg.seed + 4242)
    day_noise = rng.normal(0.0, th.ambient_daily_noise_c, size=days)

    day_mean = np.empty(days)
    day_amp = np.empty(days)
    for d in range(days):
        cur = start + _dt.timedelta(days=d)
        m = cur.month - 1                          # 0-index ay
        # ayin gunune gore komsu ayla lineer harmanlama (yumusak mevsim gecisi)
        dim = _days_in_month(cur.year, cur.month)
        frac = (cur.day - 1) / max(dim - 1, 1)     # 0..1 ay icinde konum
        nxt = (m + 1) % 12
        prv = (m - 1) % 12
        if frac < 0.5:                             # ayin ilk yarisi: onceki aya dogru
            w = 0.5 - frac
            mean = means[m] * (1 - w) + means[prv] * w
            amp = amps[m] * (1 - w) + amps[prv] * w
        else:                                      # ikinci yari: sonraki aya dogru
            w = frac - 0.5
            mean = means[m] * (1 - w) + means[nxt] * w
            amp = amps[m] * (1 - w) + amps[nxt] * w
        day_mean[d] = mean + day_noise[d]
        day_amp[d] = amp

    minute_idx = np.arange(T)
    d_of = minute_idx // MINUTES_PER_DAY
    hour = (minute_idx % MINUTES_PER_DAY) / 60.0
    intraday = np.sin(2 * np.pi * (hour - 9.0) / 24.0)         # tepe ~15:00
    theta_a = day_mean[d_of] + day_amp[d_of] * intraday
    return theta_a.astype(np.float64)


def seasonal_aging_factor(cfg: SimConfig) -> float:
    """
    MEVSIMSEL YASLANMA FAKTORU (madde 3 - 30 yil ekstrapolasyonu).

    Simulasyon penceresi EN KOTU (yaz) aylari kapsar; tum yila tasirken kis
    aylarinin DUSUK ortam sicakligi yaslanmayi ustel olarak azaltir. Faktor:
        s = <2^(θa_gun/6)>_yil / <2^(θa_gun/6)>_pencere   (< 1)
    Ankara aylik ortalamalarindan, gun-agirlikli hesaplanir (gun-ici salinim
    oran icinde buyuk olcude sadelesir; gunluk ortalama yeterli yaklasimdir).
    """
    th = cfg.thermal
    k = th.aging_doubling_k
    means = np.asarray(th.ankara_monthly_mean_c, dtype=np.float64)

    # Tum yil: her takvim gunu -> ayinin ortalamasi
    year_means = []
    for mo in range(1, 13):
        dim = _days_in_month(2025, mo)
        year_means.extend([means[mo - 1]] * dim)
    year_means = np.asarray(year_means, dtype=np.float64)
    year_metric = float(np.mean(np.power(2.0, year_means / k)))

    # Pencere: simule edilen gunlerin gun-ortalamalari
    start = _sim_start_date(cfg)
    win_means = []
    for d in range(cfg.days):
        cur = start + _dt.timedelta(days=d)
        win_means.append(means[cur.month - 1])
    win_metric = float(np.mean(np.power(2.0, np.asarray(win_means) / k)))

    return year_metric / max(win_metric, 1e-9)


def energy_seasonal_factor(cfg: SimConfig) -> float:
    """
    ENERJI TASARRUFU MEVSIM FAKTORU (N4 - yillik projeksiyon durustlugu).

    Enerji arbitraj tasarrufu gun-ici fiyat makasina baglidir. Makasin iki
    bileseni vardir:
      (a) aksam puant kacinmasi  -> yil boyu ~sabit (kis aksam puanti da yuksek),
      (b) ucuz gunduz (solar~0) penceresinin derinligi -> mevsime bagli (kis sig).
    Yaklasik vekil (proxy): proxy(ay) = 0.5 + 0.5 x solar_f(ay)  (esit agirlik).
        faktor = <proxy>_yil / <proxy>_pencere      (yaz penceresi icin < 1)
    Yillik enerji tasarrufu = donem_tasarrufu x (365/gun) x faktor. Boylece yaz
    penceresinden duz x365/gun ekstrapolasyonun iyimserligi torpulenir (termal
    kalemdeki seasonal_aging_factor'un enerji karsiligi).
    """
    pc = cfg.pricing
    msf = np.asarray(pc.monthly_solar_factor, dtype=np.float64)
    proxy = 0.5 + 0.5 * msf

    year_vals = []
    for mo in range(1, 13):
        year_vals.extend([proxy[mo - 1]] * _days_in_month(2025, mo))
    year_metric = float(np.mean(year_vals))

    start = _sim_start_date(cfg)
    win_vals = [proxy[(start + _dt.timedelta(days=int(d))).month - 1] for d in range(cfg.days)]
    win_metric = float(np.mean(win_vals))
    return year_metric / max(win_metric, 1e-9)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = _dt.date(year + 1, 1, 1)
    else:
        nxt = _dt.date(year, month + 1, 1)
    return (nxt - _dt.date(year, month, 1)).days


# --------------------------------------------------------------------------- #
# 2c) TURKIYE RESMI TATILLERI (madde 3) - baz yuk profiline girer
# --------------------------------------------------------------------------- #
# SABIT ulusal/resmi tatiller (ay, gun) - her yil ayni tarih.
_FIXED_HOLIDAYS = {
    (1, 1),    # Yilbasi
    (4, 23),   # Ulusal Egemenlik ve Cocuk Bayrami
    (5, 1),    # Emek ve Dayanisma Gunu
    (5, 19),   # Ataturk'u Anma, Genclik ve Spor Bayrami
    (7, 15),   # Demokrasi ve Milli Birlik Gunu
    (8, 30),   # Zafer Bayrami
    (10, 29),  # Cumhuriyet Bayrami
}
# DINI bayramlar (Ramazan/Kurban) her yil kayar -> yila gore (ay, gun) listesi.
# Arife gunleri dahil; 2025-2026 resmi takvimine gore.
_RELIGIOUS_HOLIDAYS = {
    2025: {
        (3, 29), (3, 30), (3, 31), (4, 1),            # Ramazan Bayrami 2025 (+arife)
        (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),       # Kurban Bayrami 2025 (+arife)
    },
    2026: {
        (3, 19), (3, 20), (3, 21), (3, 22),           # Ramazan Bayrami 2026 (+arife)
        (5, 26), (5, 27), (5, 28), (5, 29), (5, 30),  # Kurban Bayrami 2026 (+arife)
    },
}


def is_turkish_holiday(d: _dt.date) -> bool:
    """Verilen takvim gununun Turkiye resmi tatili olup olmadigi (madde 3)."""
    if (d.month, d.day) in _FIXED_HOLIDAYS:
        return True
    return (d.month, d.day) in _RELIGIOUS_HOLIDAYS.get(d.year, set())


def holiday_mask(cfg: SimConfig) -> np.ndarray:
    """Simulasyon penceresi icin gun-bazli resmi tatil maskesi (shape: days,)."""
    start = _sim_start_date(cfg)
    return np.array(
        [is_turkish_holiday(start + _dt.timedelta(days=int(d))) for d in range(cfg.days)],
        dtype=bool,
    )


def weekday_by_day(cfg: SimConfig) -> np.ndarray:
    """
    GERCEK TAKVIM hafta gunu (0=Pazartesi ... 6=Pazar), gun-bazli (shape: days,).
    Baz yuk ve oturum uretimi ARTIK sim-goreli (day%7) yerine bunu kullanir; boylece
    'pazar gunu' (==6) ve resmi tatiller GERCEK takvime hizalanir (madde 3).
    """
    start = _sim_start_date(cfg)
    return np.array(
        [(start + _dt.timedelta(days=int(d))).weekday() for d in range(cfg.days)],
        dtype=int,
    )


# --------------------------------------------------------------------------- #
# 3) Tesis BAZ YUK profili (tepe ≈ %60 trafo - madde 9)
# --------------------------------------------------------------------------- #
def generate_base_load(cfg: SimConfig) -> np.ndarray:
    """
    Tesis baz yuk profili (kW), dakika bazli. Tepe noktasi trafonun
    base_peak_frac (varsayilan %60 -> ≈912 kW @ 1520 kW etkin anma) seviyesine ulasir.

    - FABRIKA: gunduz operasyon yuksek (vardiyalar), gece dusuk.
    - AVM     : gunduz/aksam yuksek, gece dusuk; hafta sonu daha yuksek.

    Tamamen vektorize.
    """
    rng = np.random.default_rng(cfg.seed + 7)
    days = cfg.days
    T = days * MINUTES_PER_DAY
    rated = cfg.station.rated_kw
    peak = cfg.station.base_peak_frac
    is_factory = cfg.scenario.is_factory

    minute_idx = np.arange(T)
    day_of_sim = minute_idx // MINUTES_PER_DAY
    minute_of_day = minute_idx % MINUTES_PER_DAY
    hour = minute_of_day / 60.0
    # GERCEK TAKVIM hafta gunu (madde 3): hafta sonu/pazar takvime hizali olsun.
    wd_by_day = weekday_by_day(cfg)
    weekday = wd_by_day[day_of_sim]
    is_weekend = weekday >= 5
    is_sunday = weekday == 6      # PAZAR (fabrika tam kapali; cumartesi TAM gun calisir)

    # RESMI TATIL maskesi (madde 3): tatil gunlerinde baz yuk profili degisir
    # (fabrika: uretim buyuk olcude durur; AVM: ziyaret yogunlugu artar).
    is_holiday_day = holiday_mask(cfg)
    is_holiday = is_holiday_day[day_of_sim]

    if is_factory:
        # Iki vardiyali fabrika: 08-12 ve 13-18 platolari + aksam vardiyasi omzu.
        # Aksam (17-21) baz yuk hala yuksektir; filo araclari tam bu saatlerde
        # donup sarja girer -> algoritmasiz'da baz+sarj trafoyu asar (overload).
        shape = (
            0.30
            + 0.55 * np.exp(-((hour - 10.5) ** 2) / (2 * 2.2 ** 2))
            + 0.55 * np.exp(-((hour - 15.5) ** 2) / (2 * 2.2 ** 2))
            + 0.45 * np.exp(-((hour - 19.5) ** 2) / (2 * 2.0 ** 2))  # aksam vardiyasi
        )
        # CUMARTESI TAM GUN calisilir (kullanici): yalniz PAZAR uretim duser (~%45).
        weekend_mult = np.where(is_sunday, 0.45, 1.0)
        # Resmi tatil: fabrikada uretim pazardan da az (≈%40 baz seviye).
        weekend_mult = np.where(is_holiday, 0.40, weekend_mult)
    else:
        # AVM: gunduz platosu + aksam tepe.
        shape = (
            0.32
            + 0.45 * np.exp(-((hour - 15.0) ** 2) / (2 * 3.5 ** 2))
            + 0.35 * np.exp(-((hour - 19.5) ** 2) / (2 * 1.8 ** 2))
        )
        weekend_mult = np.where(is_weekend, 1.08, 1.0)
        # Resmi tatil: AVM ziyaret yogunlugu guclu hafta sonu gibi (≈+%12).
        weekend_mult = np.where(is_holiday, 1.12, weekend_mult)

    shape = shape * weekend_mult
    # Sablonu [0,1]'e olcekle, sonra tepe = peak olacak sekilde carp.
    shape = shape / shape.max()
    base_frac = shape * peak
    # Gece tabani sifirlanmasin (A3: anlamsiz cift-clip satiri kaldirildi)
    base_frac = np.clip(base_frac, 0.15, peak)

    # GUN-BAZLI DEGISKENLIK (gunler tipatip ayni olmasin): iki bilesen eklenir.
    #  (1) GUN TEPE SEVIYESI: PTF ile koreleli daily_demand_factor, [0.83, 1.0]
    #      bandina eslenir -> her gunun tepe seviyesi FARKLI ama tasarim tavanini
    #      (peak) ASMAZ; boylece optimize sarj headroom'u korunur ve baz yuk tek
    #      basina trafoyu/sozlesmeyi asmaz. Yuksek-talep gunleri tavana yakin.
    #  (2) GUN-ICI SEKIL JITTERI: her gunun profili biraz farkli (sabah/aksam
    #      oranlari gunden gune oynar) -> egriler ust uste binmez.
    df = daily_demand_factor(cfg)                       # A7: PTF korelasyonu (ortak)
    dmin, dmax = float(df.min()), float(df.max())
    day_scale = 0.83 + 0.17 * (df - dmin) / max(dmax - dmin, 1e-9)   # -> [0.83, 1.00]
    base_frac = base_frac * day_scale[day_of_sim]

    rng_v = np.random.default_rng(cfg.seed + 71)
    hour_jit = rng_v.normal(1.0, 0.06, size=(days, 24))             # gun×saat sekil oynamasi
    hour_jit = (hour_jit + np.roll(hour_jit, 1, axis=1) + np.roll(hour_jit, -1, axis=1)) / 3.0
    base_frac = base_frac * np.repeat(hour_jit.reshape(-1), 60)

    noise = rng.normal(0.0, 0.008, size=T)
    # Madde 9: baz yuk TEPESI tasarim tavanini (peak) asmaz (guvenlik kirpmasi;
    # gun-bazli degiskenlik bu tavanin ALTINDA gerceklesir).
    base_frac = np.clip(base_frac + noise, 0.10, peak)

    return (base_frac * rated).astype(np.float64)


# --------------------------------------------------------------------------- #
# 4) EPIAS PTF / SMF egrileri (TL/MWh) - saatlik, sonra dakikaya genisletilir
# --------------------------------------------------------------------------- #
def generate_market_prices(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Son `days` gune ait, 2026 EPIAS gerceklerine KALIBRE PTF/SMF egrileri uretir.

    2026 GUN-ICI PROFILI (gercek gozleme dayali, ornek 07.06.2026):
      - GECE/SABAH (00-07): ~baz seviye (~800 TL/MWh), 03-05 arasi hafif dip.
      - GUNDUZ (08-16): GUNES bollugunda PTF GENIS (yaklasik 8-10 saatlik) bir
        platoda ~0'a iner. Bu pencere DAR DEGILDIR; yuksek-solar gunlerde butun
        ogle bandi sifirlanir (eski modelin dar gaussian'i gercekci degildi).
      - AKSAM PUANT (19-21): gunes cekilip talep zirve yapinca PTF azami fiyata
        (~2700) sert bir tepe ile firlar (gunun en pahali saatleri).
      - Gunluk ortalama, solar derinligi ve baz seviye gunden gune (hava/mevsim)
        belirgin oynar; gece gunduz sifirindan PAHALIDIR.

    SMF, dengesizlik durumuna gore PTF etrafinda sapar (bazen > PTF, bazen < PTF).

    Donus: (ptf_minute, smf_minute) -> TL/MWh, shape (days*1440,)
    """
    rng = np.random.default_rng(cfg.seed + 23)
    pc = cfg.pricing
    days = cfg.days
    h = np.arange(24)

    # ---- GENIS ogle GUNES penceresi (08:00-16:00 duz plato) ----
    # Iki lojistik kapinin carpimi -> duz tabanli "boxcar". Kenarlar 07-08 ve
    # 16-17'de yumusakca acilir/kapanir. Bu, 0'a inisin GENIS olmasini saglar.
    solar_window = (
        1.0 / (1.0 + np.exp(-(h - 7.5) / 0.7))
        * 1.0 / (1.0 + np.exp((h - 16.5) / 0.7))
    )

    # ---- Bastirma ONCESI gun-ici sekil (mutlak TL/MWh) ----
    # gece bazi - gece yarisi dip + sabah omuz  (aksam puant tepe AYRI eklenir)
    base_shape = (
        pc.ptf_night_base
        - pc.ptf_night_dip * np.exp(-((h - 4.0) ** 2) / (2 * 2.5 ** 2))
        + pc.ptf_morning_bump * np.exp(-((h - 8.0) ** 2) / (2 * 1.5 ** 2))
    )
    # Aksam puant tepe (~20:00) -> azami fiyat civari. Solar bastirmadan
    # etkilenmez (o saatte solar_window ~0).
    peak_shape = (pc.ptf_evening_peak - pc.ptf_night_base) * np.exp(
        -((h - 20.0) ** 2) / (2 * 1.7 ** 2)
    )

    # ---- Gunluk degiskenlik (hava/mevsim/talep) ----
    # A7: ortak TALEP/MEVSIM faktoru ile olcekle -> baz yuk ile PTF KORELELI olur
    # (yuksek-talep/mevsim gunlerinde fiyat da yuksek). Uzerine bagimsiz gurultu.
    demand_factor = daily_demand_factor(cfg)
    day_level = np.clip(demand_factor * rng.normal(1.0, 0.06, size=days), 0.5, 1.7)
    weekday = np.arange(days) % 7
    day_level = day_level * np.where(weekday >= 5, 0.90, 1.0)         # haftasonu ucuz
    peak_daily = np.clip(demand_factor * rng.normal(1.0, 0.08, size=days), 0.6, 1.4)  # aksam tepe
    # solar derinligi: 2026'da cogu gun yuksek-solar; bazen bulutlu (dusuk).
    solar_depth = np.clip(
        rng.beta(2.4, 1.3, size=days) * (pc.ptf_solar_max - pc.ptf_solar_min)
        + pc.ptf_solar_min, 0.0, 1.0,
    )
    # AYLIK SOLAR FAKTORU (N4): gunduz ~0 penceresinin derinligi mevsime baglidir
    # (kis aylarinda gunes zayif -> bastirma sig). Gun, GERCEK takvim ayina
    # eslenir; boylece 12-aylik simulasyonlar gercek mevsimselligi tasir.
    start = _sim_start_date(cfg)
    msf = np.asarray(pc.monthly_solar_factor, dtype=np.float64)
    month_factor = np.array(
        [msf[(start + _dt.timedelta(days=int(d))).month - 1] for d in range(days)]
    )
    solar_depth = np.clip(solar_depth * month_factor, 0.0, 1.0)

    # ---- Saatlik matris (days x 24) ----
    ptf_hourly = (
        base_shape[None, :] * day_level[:, None]
        + peak_shape[None, :] * peak_daily[:, None]
    )
    # GENIS ogle penceresinde GUNES bastirmasi -> yuksek-solar gunde gunduz ~0
    suppression = 1.0 - solar_depth[:, None] * solar_window[None, :]
    ptf_hourly = ptf_hourly * np.clip(suppression, 0.0, 1.0)
    ptf_hourly = ptf_hourly + rng.normal(0.0, 60.0, size=(days, 24))   # saatlik gurultu
    ptf_hourly = np.clip(ptf_hourly, pc.ptf_floor, pc.ptf_cap)

    # ---- SMF: dengesizlik yonune gore PTF etrafinda sapar ----
    # NOT (A6): SMF yalnizca REFERANS/bilgi amaclidir; maliyet hesabi PTF (veya
    # 3-zamanli tarife) uzerinden yapilir. SMF, dengesizlik maliyeti modeline
    # dahil DEGILDIR (dengeden sorumlu taraf degiliz varsayimi).
    imbalance = rng.normal(0.0, 1.0, size=(days, 24))
    smf_hourly = ptf_hourly + pc.smf_spread * np.tanh(imbalance)
    smf_hourly = np.clip(smf_hourly, pc.ptf_floor, pc.ptf_cap + pc.smf_spread)

    # ---- Saatlik -> dakikaya genislet (her saat 60 dakika) ----
    ptf_min = np.repeat(ptf_hourly.reshape(-1), 60)       # (days*24*60,)
    smf_min = np.repeat(smf_hourly.reshape(-1), 60)
    return ptf_min.astype(np.float64), smf_min.astype(np.float64)


# --------------------------------------------------------------------------- #
# 5) STOKASTIK sarj oturumlari (kalici filo uzerinden)
# --------------------------------------------------------------------------- #
def generate_sessions(cfg: SimConfig, fleet: pd.DataFrame) -> pd.DataFrame:
    """
    Kalici filo araclari icin 100 gunluk sarj oturumlarini uretir.
    Her (arac, gun) ikilisi icin, gunluk sarj olasiligina gore bir oturum acilir.

    STOKASTIK SoC: her arac her gun FARKLI bir gelis SoC'si ile gelir.
    """
    rng = np.random.default_rng(cfg.seed + 5)
    sc = cfg.scenario
    days = cfg.days
    target = cfg.station.target_soc
    n = len(fleet)

    # GERCEK TAKVIM hafta gunu + resmi tatil (madde 3):
    #   - PAZAR (==6) ve RESMI TATIL: hicbir arac sarj OLMAZ (depo/tesis kapali).
    #   - PAZAR + RESMI TATIL HARICI (Pzt-Cumartesi): BUTUN araclar girer, TAM GUN
    #     vardiya (~17:30 doner ve fise takilir). Cumartesi de tam gun calisilir.
    wd_by_day = weekday_by_day(cfg)              # 0=Pzt..6=Paz (shape: days,)
    hol_by_day = holiday_mask(cfg)               # resmi tatil (shape: days,)
    sun_by_day = wd_by_day == 6                   # PAZAR
    sat_by_day = wd_by_day == 5                   # CUMARTESI (yarim gun)
    nocharge_by_day = sun_by_day | hol_by_day     # PAZAR + RESMI TATIL -> sarj YOK

    # (arac x gun) tam carpim, sonra GUN-BAZLI olasilikla filtrele -> vektorize
    veh_idx = np.repeat(np.arange(n), days)
    day_id = np.tile(np.arange(days), n)
    weekday = wd_by_day[day_id]
    is_weekend = weekday >= 5
    is_sunday = sun_by_day[day_id]
    is_saturday = sat_by_day[day_id]
    is_holiday = hol_by_day[day_id]
    is_nocharge = nocharge_by_day[day_id]         # pazar veya resmi tatil

    # GUN-BAZLI sarj olasiligi:
    #   - pazar + tatil: 0 (hic arac girmez)
    #   - FABRIKA: Pzt-Cumartesi -> 1.0 (BUTUN araclar girer, tam gun)
    #   - AVM: Pzt-Cumartesi -> stokastik (musteri), pazar/tatil 0
    base_prob = sc.daily_charge_prob()
    if sc.is_factory:
        prob_arr = np.where(is_nocharge, 0.0, 1.0)
    else:
        prob_arr = np.where(is_nocharge, 0.0, base_prob)
    keep = rng.random(veh_idx.shape[0]) < prob_arr
    veh_idx = veh_idx[keep]; day_id = day_id[keep]
    weekday = weekday[keep]; is_weekend = is_weekend[keep]
    is_sunday = is_sunday[keep]; is_saturday = is_saturday[keep]
    is_holiday = is_holiday[keep]
    m = veh_idx.shape[0]

    capacity = fleet["capacity"].values[veh_idx]
    dc_max = fleet["dc_max"].values[veh_idx]

    # Stokastik gelis SoC
    if sc.is_factory:
        mean_soc = np.where(is_weekend, 0.32, 0.24)
        sd_soc = 0.07
    else:
        mean_soc = np.where(is_weekend, 0.42, 0.38)
        sd_soc = 0.10
    arrival_soc = np.clip(rng.normal(mean_soc, sd_soc), 0.05, target - 0.05)

    # Gelis dakikasi ve deadline
    if sc.is_factory:
        # Pzt-Cumartesi (tam gun): tum filo TAM vardiya sonunda (~17:30) ESZAMANLI
        # doner ve fise takilir (depo surge). (Pazar/tatil gunlerinde oturum yok.)
        arr_min = np.clip(rng.normal(17.5 * 60, 0.5 * 60, size=m), 16 * 60 + 30, 19 * 60).astype(int)
        # Cikis: ertesi sabah 05:00-07:00 (gece bekleyebilir, delay cap yok).
        dep_off = rng.integers(5 * 60, 7 * 60, size=m)
        dep_global = day_id * MINUTES_PER_DAY + MINUTES_PER_DAY + dep_off
    else:
        # AVM: 09:00-20:00 gelis (Pzt-Cumartesi tam gun). Pazar/tatil yok.
        arr_min = np.clip(rng.normal(15.0 * 60, 2.5 * 60, size=m), 9 * 60, 20 * 60).astype(int)
        stay = np.clip(
            rng.normal(0.7 * sc.avm_max_stay_min, 0.25 * sc.avm_max_stay_min, size=m),
            45, sc.avm_max_stay_min,
        ).astype(int)
        dep_local = np.minimum(arr_min + stay, 22 * 60 + 30)
        dep_global = day_id * MINUTES_PER_DAY + dep_local

    arr_global = day_id * MINUTES_PER_DAY + arr_min
    energy_need = np.maximum(0.0, (target - arrival_soc) * capacity)

    sessions = pd.DataFrame({
        "session_id": np.arange(m),
        "vehicle_id": fleet["vehicle_id"].values[veh_idx],
        "model": fleet["model"].values[veh_idx],
        "segment": fleet["segment"].values[veh_idx],
        "day": day_id,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "is_saturday": is_saturday,
        "is_sunday": is_sunday,
        "is_holiday": is_holiday,
        "capacity": capacity,
        "dc_max": dc_max,
        "arrival_soc": arrival_soc,
        "target_soc": target,
        "arr_min": arr_min,
        "arr_global": arr_global.astype(int),
        "dep_global": dep_global.astype(int),
        "energy_need_kwh": energy_need,
    })
    sessions = sessions.sort_values("arr_global").reset_index(drop=True)
    return sessions


# --------------------------------------------------------------------------- #
# 6) Diske kaydet / tek-cagri uret
# --------------------------------------------------------------------------- #
def _data_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(here, "data")
    os.makedirs(d, exist_ok=True)
    return d


def save_dataset(fleet, sessions, base_load, ptf, smf, tag="FABRIKA"):
    """Veri setini data/ klasorune kaydeder."""
    d = _data_dir()
    market = pd.DataFrame({"minute": np.arange(len(ptf)), "ptf_tl_mwh": ptf, "smf_tl_mwh": smf})
    base_df = pd.DataFrame({"minute": np.arange(len(base_load)), "base_load_kw": base_load})
    paths = {
        "fleet": os.path.join(d, f"fleet_{tag}.parquet"),
        "sessions": os.path.join(d, f"sessions_{tag}.parquet"),
        "base": os.path.join(d, f"base_load_{tag}.parquet"),
        "market": os.path.join(d, f"market_{tag}.parquet"),
    }
    try:
        fleet.to_parquet(paths["fleet"], index=False)
        sessions.to_parquet(paths["sessions"], index=False)
        base_df.to_parquet(paths["base"], index=False)
        market.to_parquet(paths["market"], index=False)
    except Exception:  # pyarrow yoksa CSV
        for k in paths:
            paths[k] = paths[k].replace(".parquet", ".csv")
        fleet.to_csv(paths["fleet"], index=False)
        sessions.to_csv(paths["sessions"], index=False)
        base_df.to_csv(paths["base"], index=False)
        market.to_csv(paths["market"], index=False)
    return paths


def build_dataset(cfg: SimConfig, save: bool = True):
    """Tek cagri ile fleet + sessions + base_load + (ptf,smf) uretir."""
    fleet = build_fleet(cfg)
    sessions = generate_sessions(cfg, fleet)
    base_load = generate_base_load(cfg)
    ptf, smf = generate_market_prices(cfg)
    if save:
        save_dataset(fleet, sessions, base_load, ptf, smf, tag=cfg.scenario.name)
    return fleet, sessions, base_load, ptf, smf


if __name__ == "__main__":
    for name in ("FABRIKA", "AVM"):
        cfg = SimConfig(days=10, scenario=ScenarioConfig(name=name))
        fleet, sessions, base, ptf, smf = build_dataset(cfg, save=True)
        print(f"[{name}] filo={len(fleet)} | oturum={len(sessions)} "
              f"| baz tepe={base.max():.0f} kW | PTF ort={ptf.mean():.0f} TL/MWh")
