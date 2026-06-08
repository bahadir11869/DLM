# -*- coding: utf-8 -*-
"""
app.py
======
Ticari (B2B) Dinamik Yuk Dagilimi, Enerji & Maliyet Optimizasyonu Dashboard'u.

Calistirma:
    pip install -r requirements.txt
    python -m streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import datetime as _dt

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SimConfig, StationConfig, PricingConfig, ThermalConfig,
    FinancialConfig, ScenarioConfig, Weights, MINUTES_PER_DAY, HOURS_PER_MINUTE,
)
from src.data_generator import build_dataset, VEHICLE_DB
from src.optimizer import run_both
from src.financials import build_price_signal, full_analysis, summarize_costs
from src import epias

st.set_page_config(page_title="DLM | DC Sarj Optimizasyonu", layout="wide", page_icon="⚡")
# EN KOTU SENARYO (madde 1): 100 gun MAYIS basindan baslar -> ~9 Agustos
# (Ankara'nin en sicak bandi -> en yuksek trafo termal yaslanmasi).
START_DATE = _dt.date(2025, 5, 1)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Onbellekli simulasyon
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_simulation(scenario, pricing_mode, days, seed, regen_token, fleet_size,
                   n200, n180, n120, alpha, beta, gamma, activation_mode,
                   use_epias, epias_user, epias_pass, base_mult,
                   pf=0.95, charge_eff=0.93, diversity=0.60):
    eff_seed = int(seed) + int(regen_token) * 7919
    cfg = SimConfig(
        days=days, seed=eff_seed,
        station=StationConfig(n_socket_200=n200, n_socket_180=n180, n_socket_120=n120,
                              power_factor=float(pf), charge_efficiency=float(charge_eff),
                              diversity_factor=float(diversity)),
        pricing=PricingConfig(mode=pricing_mode, use_epias=use_epias,
                              epias_username=epias_user, epias_password=epias_pass),
        thermal=ThermalConfig(), financial=FinancialConfig(),
        scenario=ScenarioConfig(name=scenario, fleet_size_override=fleet_size),
        weights=Weights(alpha=alpha, beta=beta, gamma=gamma),
        activation_mode=activation_mode,
    )
    fleet, sessions, base_load, ptf, smf = build_dataset(cfg, save=True)

    # Senaryo-7: baz yuk carpani (1.0 normal, 1.15 = +%15)
    base_load = base_load * float(base_mult)

    # PTF kaynagi: EPIAS canli (mode=PTF ve use_epias) -> aksi halde sentetik
    ptf_source = "Sentetik (modellenmis EPIAS profili)"
    if pricing_mode == "PTF" and use_epias:
        real_ptf, src = epias.get_ptf_minute(days, None, epias_user, epias_pass, DATA_DIR)
        if real_ptf is not None and len(real_ptf) == len(ptf):
            ptf = real_ptf
            ptf_source = src
        else:
            ptf_source = f"Sentetik (EPIAS basarisiz: {src})"

    price = build_price_signal(cfg, ptf, smf)
    results = run_both(cfg, sessions, base_load, price, cfg.weights)
    analysis = full_analysis(cfg, fleet, results["optimized"], results["naive"])

    # ARAC BAZLI GUNLUK SARJ SURELERI (madde 8): her oturum icin algoritma oncesi
    # (naive) ve sonrasi (opt) sarj suresi - gun bazli filtrelenebilir.
    so = results["optimized"].sessions[["session_id", "day", "vehicle_id", "model",
                                        "charge_duration_min", "completed"]]
    sn = results["naive"].sessions[["session_id", "charge_duration_min", "completed"]]
    sessions_cmp = so.merge(sn, on="session_id", suffixes=("_opt", "_naive"))

    return {
        "cfg_days": days, "rated_kw": cfg.station.rated_kw,
        "transformer_kva": cfg.station.transformer_kva, "power_factor": cfg.station.power_factor,
        "diversity_factor": cfg.station.diversity_factor,
        "charge_efficiency": cfg.station.charge_efficiency,
        "diversified_kw": cfg.station.diversified_demand_kw,
        "ramp_kw_per_min": cfg.station.ramp_kw_per_min,
        "pricing_mode": pricing_mode, "ptf_source": ptf_source,
        "fleet_size": len(fleet), "n_sessions": len(sessions),
        "installed_kw": float(cfg.station.installed_kw),
        "base_min_kw": float(base_load.min()), "base_max_kw": float(base_load.max()),
        "base_peak_frac": float(base_load.max() / cfg.station.rated_kw),
        "max_install_kw": float(cfg.station.rated_kw - base_load.min()),
        "sessions_cmp": sessions_cmp,
        "base_load": base_load,
        "facility_opt": results["optimized"].facility_kw,
        "facility_naive": results["naive"].facility_kw,
        "charging_opt": results["optimized"].charging_kw,
        "charging_naive": results["naive"].charging_kw,
        "price": price, "ptf": ptf, "smf": smf,
        "analysis": analysis,
        "_socket_list": cfg.station.socket_list(), "_n_sockets": cfg.station.n_sockets,
        "contracted_kw": cfg.financial.contracted_demand_kw,
        "demand_penalty_naive": analysis["costs_naive"]["demand_penalty_tl"],
        "demand_penalty_opt": analysis["costs_opt"]["demand_penalty_tl"],
        "peak_naive": analysis["costs_naive"]["peak_facility_kw"],
        "peak_opt": analysis["costs_opt"]["peak_facility_kw"],
    }


def fmt_tl(x):
    return f"{x:,.0f} ₺".replace(",", ".")


def day_roi(payload, peak_opt, peak_naive):
    sl = payload["_socket_list"]; ns = payload["_n_sockets"]
    shave = max(0.0, peak_naive - peak_opt)
    pct = (shave / peak_naive * 100.0) if peak_naive > 0 else 0.0
    avg = float(np.mean(sl)) if sl else 180.0
    extra = int(shave // avg)
    return shave, pct, extra, (extra / max(ns, 1)) * 100.0


# --------------------------------------------------------------------------- #
# Kenar cubugu
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Yapilandirma")
scenario = st.sidebar.selectbox("B2B Senaryo", ["FABRIKA", "AVM"],
    help="FABRIKA/Lojistik: filo vardiya sonu eszamanli doner, gece bekler. "
         "AVM: yuksek gunduz baz yuk, Max Delay Cap var.")
pricing_mode = st.sidebar.selectbox("Fiyatlandirma Olcegi", ["PTF", "TARIFE"],
    help="PTF: EPIAS piyasa fiyati (buyuk fabrika; oglen yenilenebilir bol -> ~0 TL/MWh). "
         "TARIFE: 3-zamanli sanayi tarifesi.")

use_epias = False; epias_user = ""; epias_pass = ""
if pricing_mode == "PTF":
    use_epias = st.sidebar.checkbox("EPIAS'tan canli PTF cek", value=False,
        help="EPIAS Seffaflik Platformu hesabinizla gercek PTF cekilir. "
             "Bos/kapaliysa modellenmis gercekci PTF kullanilir.")
    if use_epias:
        epias_user = st.sidebar.text_input("EPIAS kullanici adi (e-posta)")
        epias_pass = st.sidebar.text_input("EPIAS sifre", type="password")

days = st.sidebar.slider("Simulasyon Suresi (gun)", 10, 100, 100, step=10)

st.sidebar.markdown("#### 🚗 Arac Sayisi (Filo)")
fleet_size = st.sidebar.slider("Kalici filo buyuklugu (arac)", 5, 150,
    30 if scenario == "FABRIKA" else 80,
    help="Sisteme kayitli, 100 gun boyunca tekrar tekrar sarj olan arac sayisi.")

st.sidebar.markdown("#### 🔌 DC Sarj Istasyonlari (tek soketli)")
st.sidebar.caption("Her istasyon TEK soketlidir (ayni anda tek arac). Varsayilan kurulum "
                   "(2/1/1 = 700 kW), cesitlilik faktoruyle trafo anmasinin ~%20-30'unda kalir.")
c1, c2, c3 = st.sidebar.columns(3)
n200 = c1.number_input("200 kW", 0, 12, 2)
n180 = c2.number_input("180 kW", 0, 12, 1)
n120 = c3.number_input("120 kW", 0, 12, 1)

with st.sidebar.expander("⚙️ Gelismis: Guc Faktoru · Verim · Cesitlilik", expanded=False):
    pf = st.slider("Ortalama guc faktoru (cosφ)", 0.85, 1.00, 0.95, 0.01,
        help="Tesis ORTALAMA guc faktoru (madde 6). Trafo etkin anmasi = kVA×cosφ "
             "(1600×0.95=1520 kW). pf<1 ise trafo gorunur guc (kVA) cinsinden daha cok "
             "yuklenir; termal yaslanma artar.")
    charge_eff = st.slider("Sarj verimi (sebeke→batarya)", 0.85, 1.00, 0.93, 0.01,
        help="DC hizli sarj ORTALAMA verimi (madde 7, ~%92-95). Sebekeden cekilen "
             "(faturalanan + trafo yuku) guc = batarya gucu / verim. Dusukse maliyet ve "
             "trafo yuku artar.")
    diversity = st.slider("Cesitlilik (esZamanlilik) faktoru", 0.30, 1.00, 0.60, 0.05,
        help="Madde 6 (IEC 60364-7-722). Algoritma-oncesi (LMS yok) esZamanli istasyon "
             "talebi = bu faktor × kurulu guc. <1 ise tum soketler ayni anda tam guce "
             "ulasmaz (gercekci talep tahmini). Algoritma = aktif yuk yonetimi (LMS).")

st.sidebar.markdown("#### 🤖 Algoritma Devreye Girme")
activation_label = st.sidebar.radio("Algoritma ne zaman calissin?",
    ["Her zaman", "Sadece puant (trafo doluluk ≥ %60)"],
    help="Puant: trafo doluluk orani (baz yuk/anma) ≥ %60. Disinda sistem "
         "bodoslama (yonetimsiz) calisir.")
activation_mode = "always" if activation_label.startswith("Her") else "peak_only"

st.sidebar.markdown("#### 🎚️ Multi-Objective Agirliklar")
alpha = st.sidebar.slider("α — Sarj Suresi (hizli bitir)", 0.0, 1.0, 0.5, 0.05)
beta = st.sidebar.slider("β — SOH Koruma (dusuk C-rate)", 0.0, 1.0, 0.5, 0.05)
gamma = st.sidebar.slider("γ — Maliyet (ucuza yiklen)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("#### 🎲 Veri Uretimi")
seed = st.sidebar.number_input("Seed (tekrarlanabilirlik)", 0, 9999, 42,
    help="Rastgele uretecin tohumu. Ayni seed+parametre = ayni veri.")
if "regen" not in st.session_state:
    st.session_state["regen"] = 0
b1, b2 = st.sidebar.columns(2)
if b2.button("🔄 Yeni Veri Seti", use_container_width=True):
    st.session_state["regen"] += 1; st.session_state["_do_run"] = True
if b1.button("🚀 Calistir", type="primary", use_container_width=True):
    st.session_state["_do_run"] = True

st.title("⚡ Dinamik Yuk Dagilimi, Enerji & Maliyet Optimizasyonu")
st.caption("1600 kVA Trafo (cosφ=0.95 → 1520 kW) · Cesitlilik Faktorlu DC Istasyonlar · "
           "PTF/SMF · IEC 60076-7 Termal Model · Gercek Ankara Sicakliklari (Mayis-Agustos)")

# --------------------------------------------------------------------------- #
# Bilgi panelleri
# --------------------------------------------------------------------------- #
with st.expander("ℹ️ Algoritma Nasil Calisir? (α, β, γ ve guc paylasimi)", expanded=False):
    st.markdown(r"""
**Onemli:** Algoritma araclarin **istasyona giris/cikis saatini DEGISTIRMEZ** ve
araci baska saate TASIMAZ. Sadece, arac **kendi fis-takili penceresi icinde**
gucu zamana yayar (sarj suresini uzatip dusuk guce ceker) ve bu pencere icindeki
**ucuz dakikalara daha cok yuk bindirir**. Giris/cikis zamani sabittir.

**Her dakika:**
1. **Acil guc:** Her aracin %80'e zamaninda ulasmasi icin gereken asgari guc daima
   verilir (tamamlanma garantisi).
2. **Firsatci guc:** Her arac icin *gelecek-farkindali* fiyat sinyali — aracin
   **kalan fis-takili suresindeki ortalama fiyata** gore "su an ucuz mu?" — ile,
   γ oraninda opsiyonel sarj, **aracin kendi penceresi icindeki** ucuz dakikalara
   bindirilir (pahali dakikalarda kisilip ucuz dakikalara birakilir; arac
   baska saate tasinmaz, sadece guc profili sekillenir).
3. **Trafo + ramp:** Toplam (baz+sarj) yuk trafo/sozlesme gucunu asamaz; dakikalik
   degisim, kurulu istasyon gucunun ~%10'u ile sinirlidir (madde 4; guc-kalitesi
   yumusatmasi — IEC 61000-3-3/-11). Cihaz kendisi ISO 15118/IEC 61851 ile saniyeler
   icinde rampa yapar; sinir SAHA EMS tercihidir.
4. **ESIT OLMAYAN paylasim:** Toplam guc araclara **oncelik skoruna** gore dagitilir:
   `skor = (0.6+α)·aciliyet + 0.8·(bosluk: 1−SoC) + β·(batarya boyutu)`.
   Yani **deadline'i yakin, SoC'si dusuk** araclar onceliklidir; β buyukse buyuk
   bataryali (dusuk C-rate) araclara goreli oncelik verilir (SOH korunur).

| Katsayi | Arttirinca |
|---|---|
| **α – Sure** | Daha hizli sarj, kisa sure, tamamlanma ↑ (C-rate ↑) |
| **β – SOH** | Dusuk C-rate, batarya korunur, sure biraz uzar |
| **γ – Maliyet** | Guc, pencere icindeki ucuz dakikalara bindirilir, enerji maliyeti ↓ |

Bunlar **goreli agirliklardir**: birini arttirip digerlerini sabit tutmak, o hedefe
**daha fazla oncelik** vermek demektir; **Toplam Tasarruf** uc hedefin bileskesidir
(tek katsayiyla her zaman artmaz; ilgili tek kalemi izlerseniz monoton gorursunuz).
""")

with st.expander("📐 Finansal Formuller, Varsayimlar ve Kaynaklar (yatirimci notu)", expanded=False):
    st.markdown(r"""
Ayrintili surum: **`FINANSAL_MODEL.md`**. Ozet:

- **Enerji Maliyeti** = Σ P(t)·(1/60)·fiyat(t). PTF: TL/MWh→/1000. Tarife: EPDK 3-zamanli.
- **Demand Charge** = Σ_ay tepe_guc · 90 TL/kW/ay. **Guc Asim Cezasi** = Σ_ay
  max(0, tepe−sozlesme) · (90 × **3**). (EPDK *Tarifeler Yonetmeligi*; ceza kati
  dagitim sirketine gore degisir.)
- **Trafo Termal (IEC 60076-7):** fark-denklemi sicak-nokta θH (gercek Ankara ortam
  sicakligi, Mayis-Agustos); bagil yaslanma V=2^((θH−98)/6); Tuketilen omur=Σ V·Δt.
  **30 yil ekstrapolasyonu** (mevsimsel duzeltme) ile **Ertelenen Degisim** =
  (naive−opt 30y omur kesri)×Trafo (4.000.000 TL).
- **SOH:** stres_kWh=Σ E·(1+0.6·C-rate²); SOH kaybi=stres/(kapasite·1500)·%20;
  **Geciktirilen Degisim** = Σ (kayip farki)×kapasite×4500 TL/kWh.
- **Cesitlilik faktoru (madde 6):** IEC 60364-7-722; algoritma-oncesi esZamanli talep
  = cesitlilik × kurulu guc. **Guc faktoru** cosφ=0.95 (etkin kW=kVA×cosφ). **Verim** %93.
- **Power-Shaving:** tirasanan kW ile ilave istasyon ve EV kapasitesi.

> Tum varsayimlar `src/config.py`'de degistirilebilir. Kesin teklif icin EPDK onayli
> guncel tarife ve piyasa fiyatlari girilmelidir.
""")

# Calistirma
if st.session_state.get("_do_run"):
    with st.spinner(f"{days} gun x 1440 dakika simule ediliyor..."):
        st.session_state["params"] = dict(
            scenario=scenario, pricing_mode=pricing_mode, days=days, seed=int(seed),
            regen_token=st.session_state["regen"], fleet_size=int(fleet_size),
            n200=int(n200), n180=int(n180), n120=int(n120),
            alpha=alpha, beta=beta, gamma=gamma, activation_mode=activation_mode,
            use_epias=use_epias, epias_user=epias_user, epias_pass=epias_pass,
            pf=float(pf), charge_eff=float(charge_eff), diversity=float(diversity),
        )
        st.session_state["payload"] = run_simulation(base_mult=1.0, **st.session_state["params"])
        # Baz yuk +%15 (guc asim cezasi) senaryosu her calistirmada OTOMATIK uretilir.
        st.session_state["P15"] = run_simulation(base_mult=1.15, **st.session_state["params"])
    st.session_state["_do_run"] = False

if "payload" not in st.session_state:
    st.info("Sol panelden parametreleri ayarlayip **Calistir** butonuna basin.")
    with st.expander("📋 Gercek Arac Veritabani (Net kWh / Max DC kW)", expanded=True):
        st.dataframe(VEHICLE_DB, use_container_width=True, hide_index=True)
    st.stop()

P = st.session_state["payload"]
A = P["analysis"]
dt = HOURS_PER_MINUTE

if P["pricing_mode"] == "PTF":
    st.info(f"📡 PTF kaynagi: **{P['ptf_source']}**")

# Kurulum guardrail (madde 6 + 9): cesitlilik faktorlu talep ve overload kontrolu
installed = P["installed_kw"]; rated = P["rated_kw"]
div = P["diversity_factor"]; diversified = P["diversified_kw"]
div_pct = diversified / rated * 100.0
base_pct = P["base_peak_frac"] * 100.0
naive_peak_pct = P["peak_naive"] / rated * 100.0
gr = st.columns(4)
gr[0].metric("Trafo Etkin Anma", f"{rated:.0f} kW", f"{P['transformer_kva']:.0f} kVA × {P['power_factor']:.2f}")
gr[1].metric("Kurulu İstasyon", f"{installed:.0f} kW", f"cesitlilik {div:.2f}")
gr[2].metric("Cesitlilikli Talep", f"{diversified:.0f} kW", f"%{div_pct:.0f} trafo (hedef %20-30)")
gr[3].metric("Baz Yuk Tepe", f"{P['base_max_kw']:.0f} kW", f"%{base_pct:.0f} trafo (hedef %60)")

if naive_peak_pct <= 100.0:
    st.success(f"✅ **Overload YOK** (madde 9): algoritma ÖNCESI (bodoslama) tepe "
        f"**{P['peak_naive']:.0f} kW** = trafo anmasinin **%{naive_peak_pct:.0f}**'i ≤ %100. "
        f"Cesitlilik faktorlu istasyon talebi (%{div_pct:.0f}) + baz tepe (%{base_pct:.0f}) "
        f"trafoyu asmaz; DLM faydasi peak-shaving, maliyet, termal omur ve SOH'tur.")
else:
    st.warning(f"⚠️ Algoritma oncesi (bodoslama) tepe **{P['peak_naive']:.0f} kW** trafo "
        f"anmasini (%{naive_peak_pct:.0f}) asiyor. Istasyon sayisini azaltin veya cesitlilik "
        f"faktorunu dusurun (madde 9: algoritma oncesi bile overload olmamali).")
if not (20.0 <= div_pct <= 30.0):
    st.info(f"ℹ️ Cesitlilikli istasyon talebi trafo anmasinin **%{div_pct:.0f}**'i "
        f"(hedef aralik %20-%30). Istasyon sayisini/cesitlilik faktorunu buna gore ayarlayin.")

top = st.columns(4)
top[0].metric(f"Toplam Tasarruf ({P['cfg_days']} gun)", fmt_tl(A["total_saving_tl"]))
top[1].metric("Yillik Projeksiyon", fmt_tl(A["annual_total_saving_tl"]))
top[2].metric("Trafo Tepe (Once→Sonra)",
              f"{P['peak_naive']:.0f}→{P['peak_opt']:.0f} kW")
top[3].metric("Filo / Oturum", f"{P['fleet_size']} / {P['n_sessions']}")

# B1: Tamamlanma orani (algoritmanin araclari ac birakmadiginin kaniti) + ekstra KPI
comp_opt = A["costs_opt"]["completion_rate"] * 100.0
comp_naive = A["costs_naive"]["completion_rate"] * 100.0
k = st.columns(4)
k[0].metric("Tamamlanma — Algoritmali", f"%{comp_opt:.1f}",
            f"{comp_opt - comp_naive:+.1f} puan vs bodoslama",
            help="%80 SoC'ye ulasan oturum orani. Algoritma araclari ac BIRAKMAMALIDIR; "
                 "bu KPI optimizasyonun tamamlanmayi feda etmedigini kanitlar.")
k[1].metric("Tamamlanma — Bodoslama", f"%{comp_naive:.1f}")
k[2].metric("Sozlesme Gucu", f"{P['contracted_kw']:.0f} kW",
            help="Guc asim cezasi bu esigin uzerinde baslar. Optimize tepe bu degere cekilir.")
k[3].metric("Ort. Sarj Suresi (Algo)", f"{A['costs_opt']['avg_charge_duration_min']:.0f} dk",
            f"bodoslama {A['costs_naive']['avg_charge_duration_min']:.0f} dk",
            help="Ortalama tekil oturum sarj suresi. Madde 3 (guc tabani, S=3) ile "
                 "asiri uzamasi engellenir.")

tabA, tabB = st.tabs(["📅 BOLUM A · 1 Gunluk Mikro Analiz", "📈 BOLUM B · 100 Gunluk Makro Analiz"])

# =========================================================================== #
# BOLUM A
# =========================================================================== #
with tabA:
    cp1, cp2 = st.columns([1, 2])
    sel_date = cp1.date_input("Gun sec (takvim)",
        value=START_DATE + _dt.timedelta(days=min(2, P["cfg_days"] - 1)),
        min_value=START_DATE, max_value=START_DATE + _dt.timedelta(days=P["cfg_days"] - 1))
    day = int(np.clip((sel_date - START_DATE).days, 0, P["cfg_days"] - 1))
    wd = ["Pzt", "Sal", "Car", "Per", "Cum", "Cmt", "Paz"][(START_DATE + _dt.timedelta(days=day)).weekday()]
    cp2.markdown(f"**Secilen gun:** {day+1}. gun · {sel_date.strftime('%d.%m.%Y')} · **{wd}**")

    lo = day * MINUTES_PER_DAY; hi = lo + MINUTES_PER_DAY
    x = np.arange(MINUTES_PER_DAY) / 60.0
    base = P["base_load"][lo:hi]; fac_opt = P["facility_opt"][lo:hi]; fac_naive = P["facility_naive"][lo:hi]
    chg_opt = P["charging_opt"][lo:hi]; chg_naive = P["charging_naive"][lo:hi]; price_d = P["price"][lo:hi]

    with st.expander("🔋 Kombine Yuk ve Maliyet Egrisi", expanded=True):
        g = st.columns(6)
        show_base = g[0].checkbox("Baz Yuk", True)
        show_naive = g[1].checkbox("Algoritma Oncesi", True)
        show_opt = g[2].checkbox("Algoritma Sonrasi", True)
        show_rated = g[3].checkbox("Trafo Anma", True)
        show_contract = g[4].checkbox("Sozlesme Gucu", True)
        show_price = g[5].checkbox("Fiyat Egrisi", True)
        fig, ax = plt.subplots(figsize=(12, 5))
        if show_base:
            ax.fill_between(x, 0, base, color="#5f6368", alpha=0.45, label="Baz Yuk", zorder=1)
            ax.plot(x, base, color="#3c4043", lw=1.5, zorder=2)
        if show_naive:
            ax.plot(x, fac_naive, color="#d93025", lw=2.2, label="Algoritma Oncesi (Bodoslama)", zorder=3)
        if show_opt:
            ax.plot(x, fac_opt, color="#1e8e3e", lw=2.2, label="Algoritma Sonrasi (Optimize)", zorder=4)
        if show_rated:
            ax.axhline(P["rated_kw"], color="black", ls="--", lw=1.3,
                       label=f"Trafo Anma ({P['rated_kw']:.0f} kW)", zorder=2)
        if show_contract:
            ax.axhline(P["contracted_kw"], color="#ea8600", ls="-.", lw=1.3,
                       label=f"Sozlesme Gucu ({P['contracted_kw']:.0f} kW)", zorder=2)
        ax.set_xlabel("Saat"); ax.set_ylabel("Guc (kW)")
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2)); ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if show_price:
            ax2 = ax.twinx()
            plabel = "PTF (TL/kWh)" if P["pricing_mode"] == "PTF" else "Tarife (TL/kWh)"
            ax2.plot(x, price_d, color="#1a73e8", lw=1.8, ls=":", label=plabel, zorder=5)
            ax2.set_ylabel(plabel, color="#1a73e8"); ax2.tick_params(axis="y", labelcolor="#1a73e8")
            h2, l2 = ax2.get_legend_handles_labels(); handles += h2; labels += l2
        ax.legend(handles, labels, loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("⚡ Sarj Egrisi · Algoritma Oncesi vs Sonrasi (madde 8)", expanded=True):
        st.caption("Yalnizca SARJ gucu (baz yuk haric). Algoritma oncesi (bodoslama) "
                   "sarj, araclar gelir gelmez sicrar; algoritma sonrasi ayni enerjiyi "
                   "ramp ile yumusatip ucuz/dusuk-baz dakikalara yayar.")
        fig2, axc = plt.subplots(figsize=(12, 3.8))
        axc.fill_between(x, 0, chg_naive, color="#d93025", alpha=0.18, zorder=1)
        axc.plot(x, chg_naive, color="#d93025", lw=2.2, label="Algoritma Oncesi (sarj)", zorder=3)
        axc.fill_between(x, 0, chg_opt, color="#1e8e3e", alpha=0.18, zorder=2)
        axc.plot(x, chg_opt, color="#1e8e3e", lw=2.2, label="Algoritma Sonrasi (sarj)", zorder=4)
        axc.set_xlabel("Saat"); axc.set_ylabel("Sarj Gucu (kW)")
        axc.set_xlim(0, 24); axc.set_xticks(range(0, 25, 2)); axc.grid(alpha=0.25)
        axc.legend(fontsize=9, loc="upper left"); fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    # ---- madde 8: ARAC BAZLI SARJ SURELERI (secilen t gunu) ----
    with st.expander("🚙 Arac Bazli Sarj Sureleri — Bu Gun (Algoritma Oncesi vs Sonrasi)", expanded=True):
        st.caption(f"{day+1}. gun ({sel_date.strftime('%d.%m.%Y')}) icinde sarja giren her "
                   "aracin algoritma ONCESI (bodoslama) ve SONRASI (optimize) sarj suresi. "
                   "Algoritma, aracin KENDI fis-takili penceresi icinde gucu yayar; sure "
                   "uzayabilir ama %80 tamamlanma korunur (S-kati taban ile sinirli).")
        scmp = P["sessions_cmp"]
        day_tbl = scmp[scmp["day"] == day].copy()
        if len(day_tbl) == 0:
            st.info("Bu gun sarja giren arac yok.")
        else:
            day_tbl["Uzama (dk)"] = (day_tbl["charge_duration_min_opt"]
                                     - day_tbl["charge_duration_min_naive"])
            show = pd.DataFrame({
                "Arac ID": day_tbl["vehicle_id"].values,
                "Model": day_tbl["model"].values,
                "Sure — Oncesi (dk)": day_tbl["charge_duration_min_naive"].values,
                "Sure — Sonrasi (dk)": day_tbl["charge_duration_min_opt"].values,
                "Uzama (dk)": day_tbl["Uzama (dk)"].values,
                "Tamam (Sonrasi)": np.where(day_tbl["completed_opt"].values, "✅", "—"),
            }).sort_values("Sure — Sonrasi (dk)", ascending=False)
            dcol = st.columns(3)
            dcol[0].metric("Bu gun oturum", f"{len(day_tbl)}")
            dcol[1].metric("Ort. sure — Oncesi", f"{np.nanmean(day_tbl['charge_duration_min_naive']):.0f} dk")
            dcol[2].metric("Ort. sure — Sonrasi", f"{np.nanmean(day_tbl['charge_duration_min_opt']):.0f} dk",
                           f"{np.nanmean(day_tbl['Uzama (dk)']):+.0f} dk ort. uzama")
            st.dataframe(show.style.format({
                "Sure — Oncesi (dk)": "{:.0f}", "Sure — Sonrasi (dk)": "{:.0f}",
                "Uzama (dk)": "{:+.0f}",
            }), use_container_width=True, hide_index=True, height=320)

    cost_naive = float(np.sum(chg_naive * dt * price_d))
    cost_opt = float(np.sum(chg_opt * dt * price_d))
    saving = cost_naive - cost_opt
    saving_pct = (saving / cost_naive * 100.0) if cost_naive > 0 else 0.0
    st.markdown("#### 💰 1 Gunluk Enerji Tasarrufu")
    cc = st.columns(4)
    cc[0].metric("Maliyet — Algoritma Oncesi", fmt_tl(cost_naive))
    cc[1].metric("Maliyet — Algoritma Sonrasi", fmt_tl(cost_opt))
    cc[2].metric("Tasarruf (TL)", fmt_tl(saving))
    cc[3].metric("Tasarruf (%)", f"%{saving_pct:.1f}")

    shave, pct, extra, ev_inc = day_roi(P, float(fac_opt.max()), float(fac_naive.max()))
    st.markdown("#### 🏭 Rezerv Yuk (Power Shaving) Yatirim Getirisi")
    st.success(
        f"Trafo tepe yukunde **%{pct:.0f}** ({shave:.0f} kW) tirasama yapildi. "
        f"Milyonluk trafo yenileme yatirimi ertelenerek **{shave:.0f} kW** boşluk "
        f"(headroom) yaratildi; ilave **{extra} adet** DC istasyon entegre edilebilir, "
        f"desteklenen EV sayisi **%{ev_inc:.0f}** artirilabilir.")

# =========================================================================== #
# BOLUM B
# =========================================================================== #
with tabB:
    th_o = A["thermal_opt"]; th_n = A["thermal_naive"]; soh = A["soh"]
    days_axis = np.arange(P["cfg_days"]) + 1

    lp = A["life_proj"]; pj_o = lp["proj_opt"]; pj_n = lp["proj_naive"]
    with st.expander("🔥 Trafo Termal Omru - IEC 60076-7 (gercek Ankara sicakliklari)", expanded=True):
        st.caption(f"Sicak-nokta IEC 60076-7:2018 fark-denklemi modeliyle, GERCEK Ankara "
                   f"ortam sicakligi (tepe ~{A['ambient_peak_c']:.0f}°C; Mayis-Agustos en kotu "
                   f"pencere) altinda. Bagil yaslanma V = 2^((theta_h-98)/6) (normal kagit, 98C ref).")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tepe Sicak-Nokta (Once->Sonra)",
                  f"{th_n['theta_hs_peak']:.0f}->{th_o['theta_hs_peak']:.0f} °C",
                  help="IEC normal kagit referansi 98°C (V=1). Altinda yaslanma normalden yavastir.")
        k2.metric("100 Gunde Tuketilen (Once->Sonra)",
                  f"%{th_n['pct_life_consumed']:.4f}->%{th_o['pct_life_consumed']:.4f}",
                  help="Simule edilen 100 gunluk (en kotu) penceredeki termal omur tuketimi.")
        k3.metric("30 Yilda Tuketilen (Once->Sonra)",
                  f"%{pj_n['pct_life_horizon']:.2f}->%{pj_o['pct_life_horizon']:.2f}",
                  help=f"Madde 3: pencere yaslanmasi mevsimsel duzeltme (faktor "
                       f"{lp['seasonal_factor']:.2f}) ile 30 yila tasinir.")
        k4.metric("Ertelenen Trafo Degisim Maliyeti (30y)", fmt_tl(A["thermal_saving_tl"]),
                  help="Madde 3: naive'in opt'a gore 30 yilda FAZLA tukettigi omur kesri "
                       "x trafo maliyeti (4.000.000 TL).")
        st.info(f"🧭 **Omur uzamasi (madde 3):** Dogru boyutlandirilmis (overload'suz) bu "
                f"yuklenmede termal yaslanma cok dusuktur; trafo omru FIZIKSEL tasarim omru "
                f"(~30 yil) ile baskilanir. Termal-esdeger omur ~{pj_n['equiv_life_thermal_years']:.0f} "
                f"yil (once) -> ~{pj_o['equiv_life_thermal_years']:.0f} yil (sonra). Asil parasal "
                f"karsilik, 30 yillik ufukta korunan omur kesridir: **{fmt_tl(A['thermal_saving_tl'])}**.")
        figt, axt = plt.subplots(figsize=(12, 3.8))
        axt.plot(days_axis, th_n["cum_aging_hours_daily"], color="#d93025", lw=2, label="Algoritmasiz")
        axt.plot(days_axis, th_o["cum_aging_hours_daily"], color="#1e8e3e", lw=2, label="Algoritmali")
        axt.fill_between(days_axis, th_o["cum_aging_hours_daily"], th_n["cum_aging_hours_daily"],
                         color="#fbbc04", alpha=0.25, label="Onlenen Yaslanma")
        axt.set_xlabel("Gun"); axt.set_ylabel("Kumulatif Esdeger Yaslanma (saat)")
        axt.grid(alpha=0.25); axt.legend(fontsize=9); figt.tight_layout(); st.pyplot(figt); plt.close(figt)

    with st.expander("🔋 Batarya Sagligi (SOH) Kumulatif Analizi", expanded=True):
        s1, s2, s3 = st.columns(3)
        s1.metric("Ort. SOH Dususu — Algoritmasiz", f"%{soh['final_soh_drop_naive_pct']:.3f}")
        s2.metric("Ort. SOH Dususu — Algoritmali", f"%{soh['final_soh_drop_opt_pct']:.3f}")
        s3.metric("Geciktirilen Batarya Degisim Bedeli", fmt_tl(soh["delayed_replacement_value_tl"]))
        figs, axs = plt.subplots(figsize=(12, 3.8))
        axs.plot(days_axis, soh["soh_naive_ts"], color="#d93025", lw=2, label="Algoritmasiz")
        axs.plot(days_axis, soh["soh_opt_ts"], color="#1e8e3e", lw=2, label="Algoritmali")
        axs.fill_between(days_axis, soh["soh_opt_ts"], soh["soh_naive_ts"],
                         color="#fbbc04", alpha=0.25, label="Korunan SOH")
        axs.set_xlabel("Gun"); axs.set_ylabel("Filo Ortalama SOH (%)")
        axs.grid(alpha=0.25); axs.legend(fontsize=9); figs.tight_layout(); st.pyplot(figs); plt.close(figs)

    with st.expander("🚗 Arac Bazli Mikro Karsilastirma Tablosu", expanded=True):
        st.caption("Her arac icin SOH dususu, toplam sarj suresi, EN KÖTÜ tekil oturumun "
                   "MUTLAK uzamasi (dk) ve TOPLAM surenin yuzde uzamasi (Σopt/Σnaive). "
                   "Yuzde, toplam-bazlidir (tekil-oturum yuzdesi hizli passenger araclarda "
                   "yaniltici biçimde sisiyordu).")
        st.dataframe(
            soh["table"].style.format({
                "SOH Dususu Algoritmali (%)": "{:.4f}", "SOH Dususu Algoritmasiz (%)": "{:.4f}",
                "SOH Korunan (puan)": "{:.4f}",
                "Toplam Sarj Suresi Algoritmali (dk)": "{:,.0f}",
                "Toplam Sarj Suresi Algoritmasiz (dk)": "{:,.0f}",
                "Maks Sarj Uzatma (dk)": "{:.0f}", "Toplam Sure Uzamasi (%)": "{:.1f}",
                "Korunan Batarya Bedeli (TL)": "{:,.0f}",
            }), use_container_width=True, hide_index=True, height=430)

    # ---- Senaryo-7: Baz Yuk +%15 (her calistirmada otomatik) ----
    with st.expander("📊 Senaryo: Baz Yuk +%15 → Guc Asim Cezasi (madde 7)", expanded=True):
        st.caption("Ayni istasyon ve araclar sabit; baz yuk %15 artirildiginda mevcut "
                   "kurulumun yarattigi GUC ASIM CEZASI (EPDK) gosterilir. Bu senaryo her "
                   "simulasyonda otomatik hesaplanir.")
        if "P15" in st.session_state:
            P15 = st.session_state["P15"]
            colp = st.columns(4)
            colp[0].metric("Sozlesme Gucu", f"{P['contracted_kw']:.0f} kW")
            colp[1].metric("Tepe — Mevcut / +%15 (Bodoslama)",
                           f"{P['peak_naive']:.0f} / {P15['peak_naive']:.0f} kW")
            colp[2].metric("Guc Asim Cezasi — Mevcut (Bodoslama)", fmt_tl(P["demand_penalty_naive"]))
            colp[3].metric("Guc Asim Cezasi — +%15 (Bodoslama)", fmt_tl(P15["demand_penalty_naive"]),
                           fmt_tl(P15["demand_penalty_naive"] - P["demand_penalty_naive"]))
            extra_pen = P15["demand_penalty_naive"] - P["demand_penalty_naive"]
            st.error(
                f"Baz yuk %15 arttiginda, mevcut istasyon sayisi **bodoslama** ile "
                f"trafoyu daha cok asiyor ve guc asim cezasi **{fmt_tl(extra_pen)}** artiyor. "
                f"**Algoritma** ayni +%15 kosulunda tepeyi {P15['peak_opt']:.0f} kW'a cekerek "
                f"cezayi **{fmt_tl(P15['demand_penalty_opt'])}**'ye sinirliyor.")
            st.info("Ceza = Σ_ay max(0, tepe−sozlesme) × güç bedeli × ceza katı (EPDK).")

    st.markdown("#### 🧾 Makro Finansal Ozet")
    f = st.columns(4)
    f[0].metric("Enerji Tasarrufu", fmt_tl(A["energy_saving_tl"]))
    f[1].metric("Demand Charge Tasarrufu", fmt_tl(A["demand_saving_tl"]))
    f[2].metric("Trafo Omru (30y ertelenen)", fmt_tl(A["thermal_saving_tl"]),
                help="IEC 60076-7 termal omur; 30 yillik ufukta ertelenen trafo degisim maliyeti (madde 3).")
    f[3].metric("SOH (Batarya) Tasarrufu", fmt_tl(A["soh_saving_tl"]))

    # ---- B3: Çoklu-seed Monte Carlo (tasarruf dağılımı) ----
    with st.expander("🎲 Coklu-seed Monte Carlo (tasarruf ne kadar sansa bagli?)", expanded=False):
        st.caption("Ayni parametreler, FARKLI rastgele tohumlarla N kez kosulur; toplam "
                   "tasarrufun ortalamasi ve dagilimi gosterilir. Dar dagilim -> sonuc saglam.")
        n_runs = st.slider("Tohum (seed) sayisi", 3, 20, 8, key="mc_n")
        if st.button("🎲 Monte Carlo Calistir"):
            bp0 = dict(st.session_state["params"])
            totals, annuals, comps, peaks = [], [], [], []
            prog = st.progress(0.0, text="Simulasyonlar kosuluyor...")
            for i in range(n_runs):
                bp = dict(bp0); bp["seed"] = int(bp0["seed"]) + i * 101
                r = run_simulation(base_mult=1.0, **bp)
                an = r["analysis"]
                totals.append(an["total_saving_tl"]); annuals.append(an["annual_total_saving_tl"])
                comps.append(an["costs_opt"]["completion_rate"] * 100.0)
                peaks.append(r["peak_opt"])
                prog.progress((i + 1) / n_runs, text=f"{i+1}/{n_runs}")
            prog.empty()
            st.session_state["mc"] = dict(totals=np.array(totals), annuals=np.array(annuals),
                                          comps=np.array(comps), peaks=np.array(peaks))
        if "mc" in st.session_state:
            mc = st.session_state["mc"]
            mcc = st.columns(4)
            mcc[0].metric("Toplam Tasarruf (ort)", fmt_tl(mc["totals"].mean()),
                          f"±{fmt_tl(mc['totals'].std())} (std)")
            mcc[1].metric("Yillik (ort)", fmt_tl(mc["annuals"].mean()))
            mcc[2].metric("Tamamlanma (ort)", f"%{mc['comps'].mean():.1f}")
            mcc[3].metric("Opt Tepe (ort)", f"{mc['peaks'].mean():.0f} kW")
            cv = mc["totals"].std() / max(mc["totals"].mean(), 1e-9) * 100.0
            figm, axm = plt.subplots(figsize=(10, 3))
            axm.hist(mc["totals"], bins=min(12, len(mc["totals"])), color="#1a73e8", alpha=0.8)
            axm.axvline(mc["totals"].mean(), color="#d93025", ls="--", lw=2, label="ortalama")
            axm.set_xlabel("Toplam Tasarruf (TL)"); axm.set_ylabel("Gun/kosum sayisi")
            axm.legend(fontsize=8); axm.grid(alpha=0.25)
            figm.tight_layout(); st.pyplot(figm); plt.close(figm)
            st.info(f"Degiskenlik katsayisi (CV) = **%{cv:.1f}**. Dusukse (≈<%15) tasarruf "
                    f"sansa az bagli, sonuc saglamdir.")

    # ---- B7: Soket kurulumu karşılaştırma matrisi ----
    with st.expander("🔧 Soket Kurulumu Karsilastirma (hangi kurulum daha iyi?)", expanded=False):
        st.caption("Ayni filo/parametreyle farkli soket kombinasyonlarini yan yana kosar.")
        presets = {
            "Mevcut": (int(n200), int(n180), int(n120)),
            "Hafif (1/1/1)": (1, 1, 1),
            "Orta (2/2/2)": (2, 2, 2),
            "Agir (3/3/2)": (3, 3, 2),
        }
        if st.button("🔧 Kurulumlari Karsilastir"):
            rows = []
            prog = st.progress(0.0)
            items = list(presets.items())
            for j, (nm, (a, b, c)) in enumerate(items):
                bp = dict(st.session_state["params"]); bp["n200"], bp["n180"], bp["n120"] = a, b, c
                r = run_simulation(base_mult=1.0, **bp); an = r["analysis"]
                rows.append({
                    "Kurulum": nm, "Kurulu kW": r["installed_kw"],
                    "Opt Tepe (kW)": r["peak_opt"], "Naive Tepe (kW)": r["peak_naive"],
                    "Toplam Tasarruf (TL)": an["total_saving_tl"],
                    "Tamamlanma (%)": an["costs_opt"]["completion_rate"] * 100.0,
                })
                prog.progress((j + 1) / len(items))
            prog.empty()
            st.session_state["cmp"] = pd.DataFrame(rows)
        if "cmp" in st.session_state:
            st.dataframe(st.session_state["cmp"].style.format({
                "Kurulu kW": "{:.0f}", "Opt Tepe (kW)": "{:.0f}", "Naive Tepe (kW)": "{:.0f}",
                "Toplam Tasarruf (TL)": "{:,.0f}", "Tamamlanma (%)": "{:.1f}",
            }), use_container_width=True, hide_index=True)

    # ---- B9: PTF kalibrasyon doğrulama (2026 referans profili) ----
    with st.expander("📡 PTF Kalibrasyon Dogrulama (sentetik vs 2026 referans)", expanded=False):
        st.caption("Sentetik PTF'nin saatlik ortalama profili, 2026 gercek EPIAS profiliyle "
                   "(genis ogle ~0 + sert aksam puant) karsilastirilir.")
        ptf_arr = np.asarray(P["ptf"], dtype=float)
        n_full_days = len(ptf_arr) // 1440
        synth_hourly = ptf_arr[:n_full_days * 1440].reshape(n_full_days, 24, 60).mean(axis=2).mean(axis=0)
        # 2026 yuksek-solar gun referansi (TL/MWh, yaklasik; EPIAS gozlemine dayali)
        ref_2026 = np.array([700, 660, 600, 560, 545, 575, 690, 700, 520, 360,
                             180, 90, 60, 70, 140, 360, 620, 1000, 1700, 2350,
                             2700, 2400, 1700, 1150], dtype=float)
        hrs = np.arange(24)
        figp, axp = plt.subplots(figsize=(11, 3.4))
        axp.plot(hrs, synth_hourly, color="#1a73e8", lw=2.2, marker="o", ms=3, label="Sentetik (ort.)")
        axp.plot(hrs, ref_2026, color="#ea8600", lw=2.0, ls="--", label="2026 referans (yuksek-solar gun)")
        axp.set_xlabel("Saat"); axp.set_ylabel("PTF (TL/MWh)")
        axp.set_xticks(range(0, 24, 2)); axp.grid(alpha=0.25); axp.legend(fontsize=8)
        figp.tight_layout(); st.pyplot(figp); plt.close(figp)
        st.caption(f"Sentetik gun ort: {ptf_arr.mean():.0f} TL/MWh · ogle(11-15) ort: "
                   f"{synth_hourly[11:15].mean():.0f} · aksam(19-21) ort: {synth_hourly[19:22].mean():.0f}")

    # ---- B6: Sonuç dışa aktarım (CSV) ----
    with st.expander("💾 Sonuclari Disa Aktar (CSV)", expanded=False):
        kpi_df = pd.DataFrame([
            ("Toplam Tasarruf (TL)", A["total_saving_tl"]),
            ("Yillik Projeksiyon (TL)", A["annual_total_saving_tl"]),
            ("Enerji Tasarrufu (TL)", A["energy_saving_tl"]),
            ("Demand Charge Tasarrufu (TL)", A["demand_saving_tl"]),
            ("Trafo Omru Tasarrufu (TL)", A["thermal_saving_tl"]),
            ("SOH Tasarrufu (TL)", A["soh_saving_tl"]),
            ("Opt Tepe (kW)", P["peak_opt"]),
            ("Naive Tepe (kW)", P["peak_naive"]),
            ("Tamamlanma Opt (%)", A["costs_opt"]["completion_rate"] * 100.0),
            ("Tamamlanma Naive (%)", A["costs_naive"]["completion_rate"] * 100.0),
        ], columns=["KPI", "Deger"])
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ KPI Ozeti (CSV)", kpi_df.to_csv(index=False).encode("utf-8-sig"),
                           "kpi_ozet.csv", "text/csv", use_container_width=True)
        d2.download_button("⬇️ Arac Tablosu (CSV)", soh["table"].to_csv(index=False).encode("utf-8-sig"),
                           "arac_tablosu.csv", "text/csv", use_container_width=True)
