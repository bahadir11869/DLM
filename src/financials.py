# -*- coding: utf-8 -*-
"""
financials.py
=============
PTF/SMF entegrasyonu, Trafo Termal Omur modeli (IEEE C57.91 benzeri),
batarya SOH kumulatif analizi ve B2B ROI hesaplamalari.

Ana fonksiyonlar:
  - build_price_signal()    : PTF (MWh->kWh) veya 3-zamanli tarife -> TL/kWh dizi
  - thermal_loss_of_life()  : IEEE C57.91 sicak-nokta + omur tuketimi
  - soh_analysis()          : 100 gun kumulatif SOH (algoritmali vs algoritmasiz)
  - summarize_costs()       : enerji + demand maliyeti
  - full_analysis()         : dashboard icin tum KPI'lari toplar

Tum agir hesaplar NumPy ile vektorizedir (termal model ust-yag gecikmesi haric;
o da O(T) tek-gecisli bir IIR'dir).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import SimConfig, MINUTES_PER_DAY, HOURS_PER_MINUTE
from .optimizer import SimResult


# --------------------------------------------------------------------------- #
# 1) FIYAT SINYALI  (PTF -> kWh  veya  3-zamanli tarife)
# --------------------------------------------------------------------------- #
def build_tariff_minute(cfg: SimConfig) -> np.ndarray:
    """3-zamanli sanayi tarifesi (TL/kWh), dakika bazli. Vektorize."""
    pc = cfg.pricing
    T = cfg.total_minutes
    hour = (np.arange(T) % MINUTES_PER_DAY) // 60
    price = np.full(T, pc.price_day, dtype=np.float64)
    price[(hour >= pc.peak_start_hour) & (hour < pc.night_start_hour)] = pc.price_peak
    price[(hour >= pc.night_start_hour) | (hour < pc.day_start_hour)] = pc.price_night
    return price


def build_price_signal(cfg: SimConfig, ptf_min: np.ndarray, smf_min: np.ndarray) -> np.ndarray:
    """
    Aktif fiyatlandirma moduna gore enerji birim fiyatini (TL/kWh) dondurur.
      - mode="PTF"   : PTF (TL/MWh) -> TL/kWh donusumu (/1000). Buyuk fabrika.
      - mode="TARIFE": 3-zamanli tarife (TL/kWh). Kucuk tesis.
    """
    if cfg.pricing.mode.upper() == "PTF":
        return ptf_min / 1000.0          # MWh -> kWh
    return build_tariff_minute(cfg)


# --------------------------------------------------------------------------- #
# 2) TRAFO TERMAL OMUR MODELI  (IEEE C57.91 benzeri)
# --------------------------------------------------------------------------- #
def thermal_loss_of_life(cfg: SimConfig, facility_kw: np.ndarray) -> Dict[str, object]:
    """
    Trafonun sicak-nokta (hot-spot) sicakligini ve yalitim omru tuketimini hesaplar.

    ADIMLAR (IEEE C57.91):
      K(t)      = S(t) / S_rated                      (per-unit yuklenme)
      ΔθTO,ult  = ΔθTO,R * ((K^2*R + 1)/(R + 1))^n    (ust-yag nihai artisi)
      ΔθTO(t)   : ust-yagin gercek artisi, tau_TO zaman sabitli 1. derece gecikme
                  (oil termal ataleti) -> ΔθTO[t] = ΔθTO[t-1] + (Δt/τ)(ult - ΔθTO[t-1])
      ΔθH(t)    = ΔθH,R * K^(2*m)                     (sicak-nokta'nin yaga gore artisi)
      θH(t)     = θ_ambient(t) + ΔθTO(t) + ΔθH(t)     (sicak-nokta sicakligi)
      FAA(t)    = exp(15000/383 - 15000/(θH+273))     (yaslanma hizlandirma faktoru,
                                                       110°C referansta FAA=1)
      LoL_saat  = Σ FAA(t) * Δt                       (esdeger yaslanma saati)
      %omur     = LoL_saat / normal_life_hours * 100

    FINANSAL KARSILIK:
      Esdeger tuketilen omur saatleri -> trafo yenileme maliyetinin orani kadar
      "tuketilen sermaye". Iki senaryo farki = onlenen erken yenileme tasarrufu.
    """
    th = cfg.thermal
    rated = cfg.station.rated_kw
    T = len(facility_kw)
    dt_h = HOURS_PER_MINUTE  # saat

    # Ortam sicakligi gun-ici profili (ogleden sonra tepe ~15:00)
    hour = (np.arange(T) % MINUTES_PER_DAY) / 60.0
    ambient = th.ambient_mean_c + th.ambient_amp_c * np.sin(2 * np.pi * (hour - 9.0) / 24.0)

    # Per-unit yuklenme
    K = facility_kw / rated

    # Ust-yag nihai artisi (vektorize)
    dTO_ult = th.dtheta_to_rated * ((K ** 2 * th.R_ratio + 1.0) / (th.R_ratio + 1.0)) ** th.n_exp

    # Ust-yag gercek artisi: 1. derece IIR (oil termal ataleti). Tek gecis O(T).
    alpha = 1.0 / th.tau_to_min     # Δt(=1 dk)/τ
    dTO = np.empty(T)
    prev = dTO_ult[0]
    for t in range(T):
        prev = prev + alpha * (dTO_ult[t] - prev)
        dTO[t] = prev

    # Sicak-nokta'nin yaga gore artisi (hizli, kararli hal)
    dHS = th.dtheta_hs_rated * K ** (2 * th.m_exp)

    theta_hs = ambient + dTO + dHS

    # Yaslanma hizlandirma faktoru (vektorize)
    FAA = np.exp(15000.0 / 383.0 - 15000.0 / (theta_hs + 273.0))

    lol_hours = float(np.sum(FAA) * dt_h)                # esdeger yaslanma saati
    real_hours = T * dt_h
    pct_life = lol_hours / th.normal_life_hours * 100.0  # tuketilen omur yuzdesi
    aging_cost = lol_hours / th.normal_life_hours * th.transformer_cost_tl

    # Ortalama yaslanma hizlandirma faktoru ve ESDEGER TRAFO OMRU (yil):
    #   Bu yuklenme profili surerse trafo, FAA katına orantili olarak daha
    #   erken biter. equiv_life_years = normal_life / (8760 * avg_FAA)
    avg_faa = lol_hours / max(real_hours, 1e-9)
    equiv_life_years = th.normal_life_hours / (8760.0 * max(avg_faa, 1e-9))

    # Gunluk kumulatif yaslanma (grafik icin) - dakika dizisini gune indir
    faa_daily = FAA.reshape(-1, MINUTES_PER_DAY).sum(axis=1) * dt_h  # gun basina saat
    cum_aging_hours = np.cumsum(faa_daily)

    return {
        "K_peak": float(K.max()),
        "theta_hs_peak": float(theta_hs.max()),
        "theta_hs_mean": float(theta_hs.mean()),
        "lol_hours": lol_hours,
        "pct_life_consumed": pct_life,
        "avg_faa": avg_faa,
        "equiv_life_years": equiv_life_years,
        "aging_cost_tl": aging_cost,
        "cum_aging_hours_daily": cum_aging_hours,    # shape (days,)
        "theta_hs": theta_hs,                        # shape (T,) - grafik icin
    }


# --------------------------------------------------------------------------- #
# 3) BATARYA SOH KUMULATIF ANALIZI (100 gun, kalici filo)
# --------------------------------------------------------------------------- #
def _soh_loss_fraction(stress_thr: np.ndarray, capacity: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """
    Stres-throughput'tan SOH kayip orani (kapasite kesri):
        loss = stress_thr / (capacity * cycle_life) * eol_capacity_loss
    (stress_thr, C-rate stresini zaten icerir.)
    """
    fin = cfg.financial
    return stress_thr / (capacity * fin.cycle_life) * fin.eol_capacity_loss


def soh_analysis(cfg: SimConfig, fleet: pd.DataFrame,
                 sess_opt: pd.DataFrame, sess_naive: pd.DataFrame) -> Dict[str, object]:
    """
    Algoritmali (opt) ve algoritmasiz (naive) durumlar icin 100 gun boyunca
    kumulatif SOH dususunu hesaplar; filo ortalamasi zaman serisi + arac bazli
    karsilastirma tablosu + finansal karsilik dondurur.
    """
    fin = cfg.financial
    days = cfg.days
    cap_map = fleet.set_index("vehicle_id")["capacity"]

    def per_day_loss(sess: pd.DataFrame) -> pd.DataFrame:
        df = sess.copy()
        df["loss_frac"] = _soh_loss_fraction(
            df["stress_throughput_kwh"].values, df["capacity"].values, cfg
        )
        # (vehicle_id, day) -> gunluk kayip
        g = df.groupby(["vehicle_id", "day"])["loss_frac"].sum().reset_index()
        return g

    g_opt = per_day_loss(sess_opt)
    g_naive = per_day_loss(sess_naive)

    # Tam (vehicle x day) izgarasi -> kumulatif
    vids = fleet["vehicle_id"].values
    grid = pd.MultiIndex.from_product([vids, np.arange(days)], names=["vehicle_id", "day"])

    def cumulative_matrix(g):
        s = g.set_index(["vehicle_id", "day"])["loss_frac"].reindex(grid, fill_value=0.0)
        mat = s.values.reshape(len(vids), days)        # (n_vehicle, days)
        return np.cumsum(mat, axis=1)                  # gunluk kumulatif kayip

    cum_opt = cumulative_matrix(g_opt)       # kapasite kesri (0..1)
    cum_naive = cumulative_matrix(g_naive)

    # Filo ortalama SOH (%) zaman serisi
    soh_opt_ts = 100.0 - cum_opt.mean(axis=0) * 100.0
    soh_naive_ts = 100.0 - cum_naive.mean(axis=0) * 100.0

    # Arac bazli final kayiplar
    final_opt = cum_opt[:, -1]
    final_naive = cum_naive[:, -1]
    pack_cost = cap_map.reindex(vids).values * fin.battery_cost_tl_per_kwh

    # Toplam sarj sureleri (arac bazli)
    def total_duration(sess):
        return sess.groupby("vehicle_id")["charge_duration_min"].sum()
    dur_opt = total_duration(sess_opt).reindex(vids).fillna(0.0).values
    dur_naive = total_duration(sess_naive).reindex(vids).fillna(0.0).values

    # Maksimum sarj gecikme yuzdesi (opt vs naive, ayni session_id eslesir)
    m = sess_opt[["session_id", "vehicle_id", "charge_duration_min"]].merge(
        sess_naive[["session_id", "charge_duration_min"]],
        on="session_id", suffixes=("_opt", "_naive"),
    )
    m = m[(m["charge_duration_min_naive"] > 0) & m["charge_duration_min_opt"].notna()]
    # Session bazinda uzatma (dakika): opt suresi - naive suresi
    m["delay_min"] = m["charge_duration_min_opt"] - m["charge_duration_min_naive"]
    m["delay_pct"] = m["delay_min"] / m["charge_duration_min_naive"] * 100.0
    # Her arac icin EN COK uzatilan tekil oturum (dakika ve %)
    max_delay_min = m.groupby("vehicle_id")["delay_min"].max().reindex(vids).fillna(0.0).values
    max_delay_pct = m.groupby("vehicle_id")["delay_pct"].max().reindex(vids).fillna(0.0).values

    table = pd.DataFrame({
        "Arac ID": vids,
        "Model": fleet["model"].values,
        "SOH Dususu Algoritmali (%)": final_opt * 100.0,
        "SOH Dususu Algoritmasiz (%)": final_naive * 100.0,
        "SOH Korunan (puan)": (final_naive - final_opt) * 100.0,
        "Toplam Sarj Suresi Algoritmali (dk)": dur_opt,
        "Toplam Sarj Suresi Algoritmasiz (dk)": dur_naive,
        "Maks Sarj Uzatma (dk)": max_delay_min,
        "Maks Sarj Gecikmesi (%)": max_delay_pct,
        "Korunan Batarya Bedeli (TL)": (final_naive - final_opt) * pack_cost,
    })

    delayed_replacement_value = float(((final_naive - final_opt) * pack_cost).sum())

    return {
        "soh_opt_ts": soh_opt_ts,             # shape (days,)
        "soh_naive_ts": soh_naive_ts,         # shape (days,)
        "final_soh_drop_opt_pct": float(final_opt.mean() * 100.0),
        "final_soh_drop_naive_pct": float(final_naive.mean() * 100.0),
        "delayed_replacement_value_tl": delayed_replacement_value,
        "table": table,
    }


# --------------------------------------------------------------------------- #
# 4) Enerji + demand maliyeti
# --------------------------------------------------------------------------- #
def summarize_costs(cfg: SimConfig, res: SimResult) -> Dict[str, float]:
    """Enerji maliyeti, demand charge ve operasyonel KPI'lar (tek strateji)."""
    fin = cfg.financial
    dt = HOURS_PER_MINUTE
    days = cfg.days

    energy_kwh_min = res.charging_kw * dt
    total_energy = float(energy_kwh_min.sum())
    energy_cost = float(np.sum(energy_kwh_min * res.price_tl_kwh))

    facility = res.facility_kw
    n_months = max(1, int(np.ceil(days / 30.0)))
    monthly_peak = np.array([
        facility[mm * 30 * MINUTES_PER_DAY:(mm + 1) * 30 * MINUTES_PER_DAY].max()
        if facility[mm * 30 * MINUTES_PER_DAY:(mm + 1) * 30 * MINUTES_PER_DAY].size else 0.0
        for mm in range(n_months)
    ])
    demand_base = float((monthly_peak * fin.demand_charge_tl_per_kw).sum())
    exceed = np.maximum(0.0, monthly_peak - fin.contracted_demand_kw)
    demand_penalty = float((exceed * fin.demand_penalty_tl_per_kw).sum())

    sess = res.sessions
    return {
        "strategy": res.strategy,
        "total_energy_kwh": total_energy,
        "energy_cost_tl": energy_cost,
        "demand_base_cost_tl": demand_base,
        "demand_penalty_tl": demand_penalty,
        "demand_cost_tl": demand_base + demand_penalty,
        "peak_facility_kw": float(facility.max()),
        "completion_rate": float(sess["completed"].mean()) if len(sess) else 0.0,
        "avg_charge_duration_min": float(np.nanmean(sess["charge_duration_min"].values)) if len(sess) else float("nan"),
        "n_sessions": int(len(sess)),
    }


# --------------------------------------------------------------------------- #
# 5) Power-shaving (rezerv yuk) yatirim getirisi metni/sayilari
# --------------------------------------------------------------------------- #
def power_shaving_roi(cfg: SimConfig, peak_opt: float, peak_naive: float) -> Dict[str, object]:
    """
    Puant tirasama (peak shaving) ile yaratilan rezerv kapasitenin getirisi.
    """
    st = cfg.station
    shave_kw = max(0.0, peak_naive - peak_opt)
    shave_pct = (shave_kw / peak_naive * 100.0) if peak_naive > 0 else 0.0

    # Yaratilan headroom (rezerv yuk)
    headroom_kw = shave_kw
    # Ortalama soket gucu ile kac ilave istasyon sigar
    avg_socket = float(np.mean(st.socket_list())) if st.n_sockets else 180.0
    extra_stations = int(headroom_kw // avg_socket)
    # Desteklenen EV sayisi artisi (~istasyon basina kapasite orani)
    ev_increase_pct = (extra_stations / max(st.n_sockets, 1)) * 100.0

    text = (
        f"Trafo tepe yukunde **%{shave_pct:.0f}** ({shave_kw:.0f} kW) tirasama yapildi. "
        f"Bu sayede milyonluk trafo yenileme yatirimi ertelenerek uretim bandi "
        f"kapasitesini buyutebilecek **{headroom_kw:.0f} kW** boşluk (headroom) yaratildi. "
        f"Olusturulan rezerv yuk ile sisteme ilave **{extra_stations} adet** DC sarj "
        f"istasyonu entegre edilebilir ve desteklenen elektrikli arac sayisi "
        f"**%{ev_increase_pct:.0f}** artirilabilir."
    )
    return {
        "shave_kw": shave_kw,
        "shave_pct": shave_pct,
        "headroom_kw": headroom_kw,
        "extra_stations": extra_stations,
        "ev_increase_pct": ev_increase_pct,
        "text": text,
    }


# --------------------------------------------------------------------------- #
# 6) Hepsini toplayan analiz
# --------------------------------------------------------------------------- #
def full_analysis(cfg: SimConfig, fleet: pd.DataFrame,
                  opt: SimResult, naive: SimResult) -> Dict[str, object]:
    """Dashboard icin tum makro KPI'lari tek sozlukte toplar."""
    c_opt = summarize_costs(cfg, opt)
    c_naive = summarize_costs(cfg, naive)
    th_opt = thermal_loss_of_life(cfg, opt.facility_kw)
    th_naive = thermal_loss_of_life(cfg, naive.facility_kw)
    soh = soh_analysis(cfg, fleet, opt.sessions, naive.sessions)
    roi = power_shaving_roi(cfg, c_opt["peak_facility_kw"], c_naive["peak_facility_kw"])

    energy_saving = c_naive["energy_cost_tl"] - c_opt["energy_cost_tl"]
    demand_saving = c_naive["demand_cost_tl"] - c_opt["demand_cost_tl"]
    thermal_saving = th_naive["aging_cost_tl"] - th_opt["aging_cost_tl"]
    soh_saving = soh["delayed_replacement_value_tl"]
    total_saving = energy_saving + demand_saving + thermal_saving + soh_saving

    days = cfg.days
    annual = 365.0 / max(days, 1)
    thermal_saving_annual = thermal_saving * annual

    return {
        "costs_opt": c_opt, "costs_naive": c_naive,
        "thermal_opt": th_opt, "thermal_naive": th_naive,
        "soh": soh, "roi": roi,
        "energy_saving_tl": energy_saving,
        "demand_saving_tl": demand_saving,
        "thermal_saving_tl": thermal_saving,
        "thermal_saving_annual_tl": thermal_saving_annual,
        "soh_saving_tl": soh_saving,
        "total_saving_tl": total_saving,
        "annual_total_saving_tl": total_saving * annual,
    }
