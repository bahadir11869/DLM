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
# 3) Tesis BAZ YUK profili (tepe ≈ %70 trafo)
# --------------------------------------------------------------------------- #
def generate_base_load(cfg: SimConfig) -> np.ndarray:
    """
    Tesis baz yuk profili (kW), dakika bazli. Tepe noktasi trafonun
    base_peak_frac (varsayilan %70 -> ≈1120 kW) seviyesine ulasir.

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
    weekday = day_of_sim % 7
    is_weekend = weekday >= 5

    if is_factory:
        # Iki vardiyali fabrika: 08-12 ve 13-18 platolari + aksam vardiyasi omzu.
        # Aksam (17-21) baz yuk hala yuksektir; filo araclari tam bu saatlerde
        # donup sarja girer -> BODOSLAMA'da baz+sarj trafoyu asar (overload).
        shape = (
            0.30
            + 0.55 * np.exp(-((hour - 10.5) ** 2) / (2 * 2.2 ** 2))
            + 0.55 * np.exp(-((hour - 15.5) ** 2) / (2 * 2.2 ** 2))
            + 0.45 * np.exp(-((hour - 19.5) ** 2) / (2 * 2.0 ** 2))  # aksam vardiyasi
        )
        weekend_mult = np.where(is_weekend, 0.45, 1.0)  # hafta sonu uretim az
    else:
        # AVM: gunduz platosu + aksam tepe.
        shape = (
            0.32
            + 0.45 * np.exp(-((hour - 15.0) ** 2) / (2 * 3.5 ** 2))
            + 0.35 * np.exp(-((hour - 19.5) ** 2) / (2 * 1.8 ** 2))
        )
        weekend_mult = np.where(is_weekend, 1.08, 1.0)

    shape = shape * weekend_mult
    # Sablonu [0,1]'e olcekle, sonra tepe = peak olacak sekilde carp.
    shape = shape / shape.max()
    base_frac = shape * peak
    # Gece tabani sifirlanmasin
    base_frac = np.clip(base_frac, 0.12 * peak / peak * 0.18, peak)
    base_frac = np.clip(base_frac, 0.15, peak)

    noise = rng.normal(0.0, 0.010, size=T)
    base_frac = np.clip(base_frac + noise, 0.10, peak + 0.02)

    return (base_frac * rated).astype(np.float64)


# --------------------------------------------------------------------------- #
# 4) EPIAS PTF / SMF egrileri (TL/MWh) - saatlik, sonra dakikaya genisletilir
# --------------------------------------------------------------------------- #
def generate_market_prices(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Son `days` gune ait EPIAS gerceklerine yakin PTF ve SMF egrileri uretir.

    PTF gun-ici sekli:
      - Gece (00-06) dusuk
      - Sabah tepe (08-11)
      - Aksam puant tepe (18-21)
      - GUNDUZ (10-16) GUNES/RUZGAR yogun -> PTF ~0 TL/MWh'ye kadar DUSER.
        Yenilenebilir uretimin yuksek oldugu gunlerde ogle saatlerinde PTF
        gercekte sifira yaklasir; bu, gunluk degisken bir "solar derinligi" ile
        modellenir (her gun farkli).

    SMF, dengesizlik durumuna gore PTF etrafinda sapar (bazen > PTF, bazen < PTF).

    Donus: (ptf_minute, smf_minute) -> TL/MWh, shape (days*1440,)
    """
    rng = np.random.default_rng(cfg.seed + 23)
    pc = cfg.pricing
    days = cfg.days

    # ---- Saatlik gun-ici sekil (sabah + aksam tepe, gece dip) ----
    h = np.arange(24)
    daily_shape = (
        0.45 * np.exp(-((h - 9.0) ** 2) / (2 * 2.0 ** 2))     # sabah tepe
        + 0.85 * np.exp(-((h - 19.5) ** 2) / (2 * 2.2 ** 2))  # aksam puant
        - 0.40 * np.exp(-((h - 3.5) ** 2) / (2 * 2.5 ** 2))   # gece dip
    )

    # ---- Gunluk ortalama seviye dalgalanmasi (hava/talep) ----
    day_level = rng.normal(0.0, 0.12, size=days)         # gunler arasi ±%12
    weekday = np.arange(days) % 7
    weekend_adj = np.where(weekday >= 5, -0.08, 0.0)     # haftasonu daha ucuz

    base = pc.ptf_mean
    amp = pc.ptf_amplitude
    ptf_hourly = (
        base
        + amp * daily_shape[None, :]
        + base * (day_level[:, None] + weekend_adj[:, None])
        + rng.normal(0.0, 90.0, size=(days, 24))          # saatlik gurultu
    )

    # ---- YENILENEBILIR (SOLAR) BASKILAMA: ogle saatlerinde PTF -> ~0 ----
    #   solar_depth: her gun icin yenilenebilir yogunlugu (0=az, 1=cok gunes)
    #   solar_hours: ogle (13:00) civarinda tepe yapan 0..1 pencere
    #   carpan supp = 1 - solar_depth*solar_hours  -> yuksek-solar gunlerde
    #   ogleyin supp~0 olur ve PTF sifira yaklasir.
    solar_depth = rng.uniform(0.0, 1.0, size=days)
    solar_hours = np.exp(-((h - 13.0) ** 2) / (2 * 2.4 ** 2))
    supp = 1.0 - solar_depth[:, None] * solar_hours[None, :]
    ptf_hourly = ptf_hourly * np.clip(supp, 0.0, 1.0)
    ptf_hourly = np.clip(ptf_hourly, pc.ptf_floor, pc.ptf_cap)

    # ---- SMF: dengesizlik yonune gore PTF etrafinda sapar ----
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

    # (arac x gun) tam carpim, sonra olasilikla filtrele -> vektorize
    veh_idx = np.repeat(np.arange(n), days)
    day_id = np.tile(np.arange(days), n)
    weekday = day_id % 7
    is_weekend = weekday >= 5

    prob = sc.daily_charge_prob()
    keep = rng.random(veh_idx.shape[0]) < prob
    veh_idx = veh_idx[keep]
    day_id = day_id[keep]
    weekday = weekday[keep]
    is_weekend = is_weekend[keep]
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
        # LOJISTIK DEPO SURGE: tum filo vardiya sonunda (17:00-19:00) ESZAMANLI
        # doner ve fise takilir. 8 soket kuyrukla dolar; aksam baz yuku hala
        # yuksekken (vardiya omzu) BODOSLAMA toplam yuku trafonun USTUNE cikarir
        # ve bu durum saatlerce surer -> gercek termal asiri yuklenme (overload).
        # Araclar ertesi sabah 05:00-07:00 cikar (gece bekleyebilir, delay cap yok).
        arr_min = np.clip(rng.normal(17.5 * 60, 0.5 * 60, size=m), 16 * 60 + 30, 19 * 60).astype(int)
        dep_off = rng.integers(5 * 60, 7 * 60, size=m)
        dep_global = day_id * MINUTES_PER_DAY + MINUTES_PER_DAY + dep_off
    else:
        # AVM: 09:00-20:00 gelis; kalma Max Delay Cap ile sinirli.
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
