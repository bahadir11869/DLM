# -*- coding: utf-8 -*-
"""
financials.py
=============
PTF/SMF entegrasyonu, Trafo Termal Omur modeli (IEC 60076-7:2018),
batarya SOH kumulatif analizi ve B2B ROI hesaplamalari.

Ana fonksiyonlar:
  - build_price_signal()         : PTF (MWh->kWh) veya 3-zamanli tarife -> TL/kWh dizi
  - thermal_loss_of_life()       : IEC 60076-7 sicak-nokta + omur tuketimi (gercek ambient)
  - transformer_life_projection(): 30 yillik omur ekstrapolasyonu + ertelenen maliyet
  - soh_analysis()               : 100 gun kumulatif SOH (algoritmali vs algoritmasiz)
  - summarize_costs()            : enerji + demand maliyeti
  - full_analysis()              : dashboard icin tum KPI'lari toplar

Tum agir hesaplar NumPy ile vektorizedir (termal model ust-yag gecikmesi haric;
o da O(T) tek-gecisli bir IIR'dir).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import SimConfig, MINUTES_PER_DAY, HOURS_PER_MINUTE
from .optimizer import SimResult
from .data_generator import ambient_series, seasonal_aging_factor, energy_seasonal_factor


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
    NOT: SMF kullanilMAZ (yalnizca referans/bilgi amaclidir).
    """
    if cfg.pricing.mode.upper() == "PTF":
        return ptf_min / 1000.0          # MWh -> kWh
    return build_tariff_minute(cfg)


# --------------------------------------------------------------------------- #
# 2) TRAFO TERMAL OMUR MODELI  (IEC 60076-7:2018) - madde 1, 2, 3
# --------------------------------------------------------------------------- #
def thermal_loss_of_life(cfg: SimConfig, facility_kw: np.ndarray,
                         ambient: np.ndarray = None) -> Dict[str, object]:
    """
    Trafonun sicak-nokta (hot-spot) sicakligini ve yalitim omru tuketimini IEC
    60076-7:2018 FARK-DENKLEMI modeli (madde 8.2.2) ile hesaplar.

    ADIMLAR (IEC 60076-7):
      K(t)     = (baz+sarj)(t) / S_anma                          (p.u. yuklenme)
      Δθo,ult  = Δθor·((1 + R·K²)/(1 + R))^x                      (ust-yag nihai artisi)
      Δθo[t]   = Δθo[t-1] + (Δt/(k11·τo))·(Δθo,ult − Δθo[t-1])    (yag ataleti, τo)
      Δθh1[t]  = Δθh1[t-1] + (Δt/(k22·τw))·(k21·Δθhr·K^y − Δθh1[t-1])
      Δθh2[t]  = Δθh2[t-1] + (Δt·k22/τo)·((k21−1)·Δθhr·K^y − Δθh2[t-1])
      Δθh      = Δθh1 − Δθh2                                      (sicak-nokta gradyani)
      θh[t]    = θa(t) + Δθo[t] + Δθh[t]                          (sicak-nokta °C)
      V(t)     = 2^((θh(t) − 98)/6)        (bagil yaslanma; normal kagit, 98°C'de V=1)
      LoL_saat = Σ V(t)·Δt                                        (esdeger yaslanma saati)

    ORTAM SICAKLIGI (madde 1): θa(t), GERCEK Ankara sicaklik serisidir (Mayis
    basindan, en kotu senaryo). Disaridan verilmezse cfg'den uretilir.
    """
    th = cfg.thermal
    rated = cfg.station.rated_kw
    T = len(facility_kw)
    dt_h = HOURS_PER_MINUTE  # saat

    if ambient is None:
        ambient = ambient_series(cfg)
    ambient = np.asarray(ambient, dtype=np.float64)
    if len(ambient) != T:                       # uzunluk uyusmazsa kirp/doldur
        if len(ambient) > T:
            ambient = ambient[:T]
        else:
            ambient = np.concatenate([ambient, np.full(T - len(ambient), ambient[-1])])

    # Per-unit yuklenme ve nihai (steady-state) artislar
    K = facility_kw / rated
    dTO_ult = th.dtheta_or * ((1.0 + th.R_ratio * K ** 2) / (1.0 + th.R_ratio)) ** th.x_oil_exp
    hs_drive = th.dtheta_hr * K ** th.y_wind_exp          # K^y surucu terimi

    # Fark denklemleri (tek gecis O(T)). Δt = 1 dakika.
    a_o = 1.0 / (th.k11 * th.tau_o_min)                   # ust-yag IIR katsayisi
    a_h1 = 1.0 / (th.k22 * th.tau_w_min)                  # hot-spot 1 (sargi)
    a_h2 = th.k22 / th.tau_o_min                          # hot-spot 2 (yag gecikmesi)

    dTO = np.empty(T)
    dHS = np.empty(T)
    o_prev = dTO_ult[0]
    h1_prev = th.k21 * hs_drive[0]
    h2_prev = (th.k21 - 1.0) * hs_drive[0]
    for t in range(T):
        o_prev = o_prev + a_o * (dTO_ult[t] - o_prev)
        h1_prev = h1_prev + a_h1 * (th.k21 * hs_drive[t] - h1_prev)
        h2_prev = h2_prev + a_h2 * ((th.k21 - 1.0) * hs_drive[t] - h2_prev)
        dTO[t] = o_prev
        dHS[t] = h1_prev - h2_prev

    theta_hs = ambient + dTO + dHS

    # Bagil yaslanma hizi V (IEC normal kagit, 98°C referans): V = 2^((θh−98)/6)
    V = np.power(th.aging_base, (theta_hs - th.hs_reference_c) / th.aging_doubling_k)

    lol_hours = float(np.sum(V) * dt_h)                  # esdeger yaslanma saati (pencere)
    real_hours = T * dt_h
    pct_life = lol_hours / th.normal_life_hours * 100.0  # PENCEREDE tuketilen omur %
    aging_cost = lol_hours / th.normal_life_hours * th.transformer_cost_tl

    avg_V = lol_hours / max(real_hours, 1e-9)            # ortalama bagil yaslanma (pencere)

    # Gunluk kumulatif yaslanma (grafik icin)
    V_daily = V.reshape(-1, MINUTES_PER_DAY).sum(axis=1) * dt_h
    cum_aging_hours = np.cumsum(V_daily)

    return {
        "K_peak": float(K.max()),
        "theta_hs_peak": float(theta_hs.max()),
        "theta_hs_mean": float(theta_hs.mean()),
        "ambient_peak": float(ambient.max()),
        "lol_hours": lol_hours,                          # pencere (sim) yaslanma saati
        "pct_life_consumed": pct_life,                   # pencerede tuketilen %
        "avg_V": avg_V,
        "aging_cost_tl": aging_cost,                     # pencere yaslanma maliyeti
        "cum_aging_hours_daily": cum_aging_hours,        # shape (days,)
        "theta_hs": theta_hs,                            # shape (T,)
    }


# --------------------------------------------------------------------------- #
# 2b) 30 YILLIK OMUR EKSTRAPOLASYONU + ERTELENEN DEGISIM MALIYETI (madde 3)
# --------------------------------------------------------------------------- #
def transformer_life_projection(cfg: SimConfig, th_opt: Dict, th_naive: Dict,
                                seasonal_factor: float = None) -> Dict[str, object]:
    """
    EN KOTU 100 gunluk pencere yaslanmasini, MEVSIMSEL DUZELTME ile yila ve
    30 yila tasiyarak (madde 3):
      - termal-esdeger trafo omru (yil),
      - DLM'in sagladigi OMUR UZAMASI (yil),
      - 30 yilda tuketilen omur yuzdesi,
      - ERTELENEN trafo degisim maliyeti = (uzamanin toplam omre orani) × trafo maliyeti
    hesaplar.

    Yaklasim:
      yillik_LoL = (pencere_LoL / sim_gun) × 365 × mevsim_faktoru
      termal_esdeger_omur = normal_omur_saat / yillik_LoL
      30y_tuketilen_kesir  = yillik_LoL × 30 / normal_omur_saat
      ertelenen_maliyet    = (30y_kesir_naive − 30y_kesir_opt) × trafo_maliyeti
    """
    th = cfg.thermal
    days = cfg.days
    if seasonal_factor is None:
        seasonal_factor = seasonal_aging_factor(cfg)

    horizon = th.extrapolation_years
    norm = th.normal_life_hours

    def project(thd: Dict) -> Dict[str, float]:
        annual_lol = (thd["lol_hours"] / max(days, 1)) * 365.0 * seasonal_factor
        equiv_life_raw = norm / max(annual_lol, 1e-9)        # termal-esdeger omur (yil)
        equiv_life = min(th.design_life_years, equiv_life_raw)
        frac_horizon = annual_lol * horizon / norm           # 30 yilda tuketilen kesir
        return {
            "annual_lol_hours": annual_lol,
            "equiv_life_thermal_years": equiv_life_raw,      # kirpilmamis (termal)
            "equiv_life_years": equiv_life,                  # tasarim tavaniyla kirpik
            "frac_life_horizon": frac_horizon,               # 0..1 (30 yil)
            "pct_life_horizon": frac_horizon * 100.0,
        }

    p_opt = project(th_opt)
    p_naive = project(th_naive)

    # OMUR UZAMASI (termal-esdeger): DLM ile trafo termal omru ne kadar uzar.
    life_extension_years = p_opt["equiv_life_thermal_years"] - p_naive["equiv_life_thermal_years"]

    # ERTELENEN DEGISIM MALIYETI (madde 3): 30 yillik ufukta naive'in opt'a gore
    # FAZLA tukettigi omur kesri × trafo maliyeti. (Uzamanin toplam omre orani,
    # tipik trafo omru = design_life uzerinden de ifade edilir.)
    extra_frac_horizon = max(0.0, p_naive["frac_life_horizon"] - p_opt["frac_life_horizon"])
    deferred_replacement_tl = extra_frac_horizon * th.transformer_cost_tl

    return {
        "seasonal_factor": seasonal_factor,
        "horizon_years": horizon,
        "proj_opt": p_opt,
        "proj_naive": p_naive,
        "life_extension_years": life_extension_years,
        "extra_frac_horizon": extra_frac_horizon,
        "deferred_replacement_tl": deferred_replacement_tl,
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

    # Sarj suresi uzamasi (opt vs naive). Iki metrik:
    #   1) "Maks Sarj Uzatma (dk)" : araç başına EN KÖTÜ tekil oturumun MUTLAK uzamasi
    #      (kac dakika). Mutlak buyukluk - yorumlamasi guvenli.
    #   2) "Toplam Sure Uzamasi (%)": araç başına TOPLAM sarj suresinin yuzde uzamasi
    #      (Σopt/Σnaive − 1). TEMSILI metrik. (Eski "tekil oturum %" metrigi, hizli
    #      sarj olan passenger araclarda minik naive tabana bolundugu icin %500-600
    #      gibi YANILTICI degerler uretiyordu; toplam-bazli oran gercekci ve sinirli.)
    m = sess_opt[["session_id", "vehicle_id", "charge_duration_min"]].merge(
        sess_naive[["session_id", "charge_duration_min"]],
        on="session_id", suffixes=("_opt", "_naive"),
    )
    m = m[(m["charge_duration_min_naive"] > 0) & m["charge_duration_min_opt"].notna()]
    m["delay_min"] = m["charge_duration_min_opt"] - m["charge_duration_min_naive"]
    max_delay_min = m.groupby("vehicle_id")["delay_min"].max().reindex(vids).fillna(0.0).values
    with np.errstate(divide="ignore", invalid="ignore"):
        total_stretch_pct = np.where(dur_naive > 0, (dur_opt / np.maximum(dur_naive, 1e-9) - 1.0) * 100.0, 0.0)

    table = pd.DataFrame({
        "Arac ID": vids,
        "Model": fleet["model"].values,
        "SOH Dususu Algoritmali (%)": final_opt * 100.0,
        "SOH Dususu Algoritmasiz (%)": final_naive * 100.0,
        "SOH Korunan (puan)": (final_naive - final_opt) * 100.0,
        "Toplam Sarj Suresi Algoritmali (dk)": dur_opt,
        "Toplam Sarj Suresi Algoritmasiz (dk)": dur_naive,
        "Maks Sarj Uzatma (dk)": max_delay_min,
        "Toplam Sure Uzamasi (%)": total_stretch_pct,
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

    energy_kwh_min = res.charging_kw * dt
    total_energy = float(energy_kwh_min.sum())
    energy_cost = float(np.sum(energy_kwh_min * res.price_tl_kwh))

    facility = res.facility_kw
    # TEPE OLCUMU (N3): EPDK/OSOS sayaclari tepe gucu 15 DAKIKALIK ORTALAMA
    # uzerinden olcer (demand_interval_min). Dakikalik anlik tepe yerine bu
    # ortalama kullanilir; kisa sivri tepeler torpulenir, ceza/tasarruf sismez.
    iv = max(1, int(fin.demand_interval_min))
    n_iv = len(facility) // iv
    if n_iv > 0:
        fac_iv = facility[:n_iv * iv].reshape(n_iv, iv).mean(axis=1)
    else:
        fac_iv = facility
    peak_iv_overall = float(fac_iv.max()) if len(fac_iv) else 0.0

    # AYLIK demand/ceza: her 30-gunluk blok; KISMI son ay gun orani (frac) kadar
    # agirlanir (yillik projeksiyon x365/gun ile tutarli kalir).
    #
    # GUC BEDELI REJIMI (N2):
    #   EPDK   -> guc bedeli = SOZLESME GUCU x birim bedel (SABIT; olculen tepeden
    #             bagimsiz -> iki stratejide AYNI, tasarruf yalnizca CEZA farkidir.
    #             Tepe dususunun ikinci getirisi "daha dusuk sozlesme gucu
    #             secebilme"dir; o ayri kalemde raporlanir).
    #   DEMAND -> olculen aylik 15-dk tepe x birim bedel (ABD-tarzi, eski davranis).
    epdk_mode = str(fin.billing_mode).upper() != "DEMAND"
    contracted = float(fin.contracted_demand_kw)
    block_iv = (30 * MINUTES_PER_DAY) // iv
    n_blocks = max(1, int(np.ceil(len(fac_iv) / block_iv)))
    demand_base = 0.0
    demand_penalty = 0.0
    for mm in range(n_blocks):
        seg = fac_iv[mm * block_iv:(mm + 1) * block_iv]
        if seg.size == 0:
            continue
        pk = float(seg.max())                # aylik 15-dk ortalama tepe
        frac = seg.size / block_iv           # kismi ay agirligi (0..1)
        if epdk_mode:
            demand_base += contracted * fin.demand_charge_tl_per_kw * frac
        else:
            demand_base += pk * fin.demand_charge_tl_per_kw * frac
        demand_penalty += max(0.0, pk - contracted) * fin.demand_penalty_tl_per_kw * frac
    demand_base = float(demand_base)
    demand_penalty = float(demand_penalty)

    sess = res.sessions
    return {
        "strategy": res.strategy,
        "total_energy_kwh": total_energy,
        "energy_cost_tl": energy_cost,
        "demand_base_cost_tl": demand_base,
        "demand_penalty_tl": demand_penalty,
        "demand_cost_tl": demand_base + demand_penalty,
        "billing_mode": "EPDK" if epdk_mode else "DEMAND",
        "peak_facility_kw": float(facility.max()),
        "peak_15min_kw": peak_iv_overall,
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

    # N6: anlatim, modelin kendi sonucuyla hizali tutulur. Dogru boyutlandirilmis
    # (overload'suz) sistemde TERMAL omur kazanci kucuktur; trafo-tarafi parasal
    # deger ASIM CEZASININ onlenmesi + dusuk sozlesme gucu + BUYUME headroom'udur.
    # Ilave istasyon/EV rakamlari, yeni talebin de DLM tavani ALTINDA yonetilmesi
    # kosuluna baglidir (kosulsuz "bedava kapasite" degildir).
    def _fmtn(x, d=0):
        s = f"{float(x):,.{d}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    text = (
        f"Trafo tepe yukunde **%{_fmtn(shave_pct)}** ({_fmtn(shave_kw)} kW) tirasama yapildi; "
        f"ayni trafoda **{_fmtn(headroom_kw)} kW** buyume payi (headroom) olustu. Bu pay, "
        f"yeni talebin de DLM tavani altinda yonetilmesi KOSULUYLA ilave "
        f"**{extra_stations} adet** DC istasyona (veya esdeger baz yuk buyumesine) alan "
        f"acar; desteklenen EV kapasitesi **%{_fmtn(ev_increase_pct)}** artirilabilir. "
        f"Trafo-tarafi parasal kazanc, guc asim cezalarinin onlenmesi ve sozlesme "
        f"gucunun dusuk tutulabilmesidir (bu boyutlandirmada termal omur kazanci "
        f"kucuktur; ayrintisi termal bolumde)."
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
    # Gercek Ankara ortam sicakligi (madde 1) - iki senaryoda da AYNI seri.
    ambient = ambient_series(cfg)
    th_opt = thermal_loss_of_life(cfg, opt.facility_kw, ambient)
    th_naive = thermal_loss_of_life(cfg, naive.facility_kw, ambient)
    # 30 yillik omur projeksiyonu + ertelenen degisim maliyeti (madde 3)
    life_proj = transformer_life_projection(cfg, th_opt, th_naive)
    soh = soh_analysis(cfg, fleet, opt.sessions, naive.sessions)
    roi = power_shaving_roi(cfg, c_opt["peak_facility_kw"], c_naive["peak_facility_kw"])

    energy_saving = c_naive["energy_cost_tl"] - c_opt["energy_cost_tl"]
    demand_saving = c_naive["demand_cost_tl"] - c_opt["demand_cost_tl"]
    # TRAFO OMUR TASARRUFU (madde 2): bu SIMULASYON DONEMINDE tuketilen omrun
    # algoritma ONCESI - SONRASI farki, 30 yillik omur butcesine (262.800 saat)
    # oranlanip TRAFO MALIYETI ile carpilir:
    #   tasarruf = (tuketilen_omur%_naive − tuketilen_omur%_opt) × trafo_maliyeti
    # (aging_cost_tl zaten pencere_lol/normal_life × trafo_maliyeti'dir.)
    thermal_saving = th_naive["aging_cost_tl"] - th_opt["aging_cost_tl"]
    thermal_saving_deferred = life_proj["deferred_replacement_tl"]   # 30y projeksiyon (referans)

    # SOH ROI ATAMASI (N5): batarya kimin mali? FABRIKA/filo senaryosunda araclar
    # tesis sahibinindir -> SOH kazanci OPERATOR ROI'sine girer. AVM senaryosunda
    # bataryalar MUSTERININDIR -> operator toplamina YAZILMAZ; ayri "musteri
    # faydasi" kalemi olarak raporlanir. (SOH modeli ad-hoc k=0.6 katsayisiyla
    # GOSTERGE niteligindedir; kalibrasyon yol haritasi Faz 3.)
    soh_saving = soh["delayed_replacement_value_tl"]
    soh_in_operator_roi = bool(cfg.scenario.is_factory)
    operator_soh_saving = soh_saving if soh_in_operator_roi else 0.0
    total_saving = energy_saving + demand_saving + thermal_saving + operator_soh_saving

    days = cfg.days
    annual = 365.0 / max(days, 1)

    # YILLIK PROJEKSIYON MEVSIM DUZELTMELERI (N4): yaz penceresinden duz
    # x365/gun ekstrapolasyon iyimserdir. Kalem bazinda duzeltme:
    #   enerji -> energy_seasonal_factor (solar makasi kisin sigdir),
    #   termal -> seasonal_aging_factor (kis ortaminda V ustel kuculur),
    #   demand/ceza -> aylik yapi; lineer birakildi,
    #   SOH -> throughput surumlu, mevsimden ~bagimsiz; lineer.
    f_energy = energy_seasonal_factor(cfg)
    f_thermal = life_proj["seasonal_factor"]
    annual_energy = energy_saving * annual * f_energy
    annual_demand = demand_saving * annual
    annual_thermal = thermal_saving * annual * f_thermal
    annual_soh = operator_soh_saving * annual
    annual_total = annual_energy + annual_demand + annual_thermal + annual_soh

    return {
        "costs_opt": c_opt, "costs_naive": c_naive,
        "thermal_opt": th_opt, "thermal_naive": th_naive,
        "life_proj": life_proj,
        "ambient_peak_c": float(np.max(ambient)),
        "soh": soh, "roi": roi,
        "energy_saving_tl": energy_saving,
        "demand_saving_tl": demand_saving,
        "thermal_saving_tl": thermal_saving,                 # bu donem omur tuketim farki × maliyet
        "thermal_saving_deferred_tl": thermal_saving_deferred,  # 30y projeksiyon (referans)
        "thermal_life_extension_years": life_proj["life_extension_years"],
        "soh_saving_tl": soh_saving,                         # ham SOH kazanci (gosterge)
        "soh_in_operator_roi": soh_in_operator_roi,          # N5: operator ROI'sine dahil mi
        "operator_soh_saving_tl": operator_soh_saving,
        "total_saving_tl": total_saving,
        "energy_seasonal_factor": f_energy,
        "annual_energy_saving_tl": annual_energy,
        "annual_demand_saving_tl": annual_demand,
        "annual_thermal_saving_tl": annual_thermal,
        "annual_soh_saving_tl": annual_soh,
        "annual_total_saving_tl": annual_total,
    }
