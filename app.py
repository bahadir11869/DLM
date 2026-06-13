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
import plotly.graph_objects as go

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
                   pf=0.95, charge_eff=0.93, diversity=0.60,
                   billing_mode="EPDK", naive_mode="diversity", dep_unc=0.0):
    eff_seed = int(seed) + int(regen_token) * 7919
    cfg = SimConfig(
        days=days, seed=eff_seed,
        station=StationConfig(n_socket_200=n200, n_socket_180=n180, n_socket_120=n120,
                              power_factor=float(pf), charge_efficiency=float(charge_eff),
                              diversity_factor=float(diversity),
                              naive_mode=str(naive_mode)),
        pricing=PricingConfig(mode=pricing_mode, use_epias=use_epias,
                              epias_username=epias_user, epias_password=epias_pass),
        thermal=ThermalConfig(), financial=FinancialConfig(billing_mode=str(billing_mode)),
        scenario=ScenarioConfig(name=scenario, fleet_size_override=fleet_size,
                                dep_uncertainty_min=float(dep_unc)),
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
        "demand_charge_tl_per_kw": float(cfg.financial.demand_charge_tl_per_kw),
        "demand_penalty_mult": float(cfg.financial.demand_penalty_multiplier),
        "demand_penalty_naive": analysis["costs_naive"]["demand_penalty_tl"],
        "demand_penalty_opt": analysis["costs_opt"]["demand_penalty_tl"],
        "peak_naive": analysis["costs_naive"]["peak_facility_kw"],
        "peak_opt": analysis["costs_opt"]["peak_facility_kw"],
        "transformer_cost_tl": float(cfg.thermal.transformer_cost_tl),
        "design_life_years": float(cfg.thermal.design_life_years),
        "usd_try_rate": float(cfg.financial.usd_try_rate),
        "billing_mode": str(billing_mode),
        "naive_mode": str(naive_mode),
        "dep_unc": float(dep_unc),
        "peak15_naive": analysis["costs_naive"]["peak_15min_kw"],
        "peak15_opt": analysis["costs_opt"]["peak_15min_kw"],
        "is_factory": bool(cfg.scenario.is_factory),
    }


def fmt_tl(x):
    return f"{x:,.0f} ₺".replace(",", ".")


def fmt_usd(x):
    return f"${x:,.0f}".replace(",", ".")


def fmt_tl_usd(x_tl, rate):
    """TL tutari hem TL hem de USD olarak gosterir (madde 2)."""
    return f"{fmt_tl(x_tl)} ({fmt_usd(x_tl / max(rate, 1e-9))})"


def fmt_num(x, decimals=0, sign=False):
    """Sayiyi Turkce formatta dondurur: binlik ayiraci nokta, ondalik ayiraci virgul."""
    if sign:
        s = f"{float(x):+,.{decimals}f}"
    else:
        s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _tr(fmt_str):
    """Pandas style.format() icin Turkce sayi formatlayici (binlik=nokta, ondalik=virgul)."""
    def _f(x):
        try:
            s = f"{float(x):{fmt_str}}"
            return s.replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)
    return _f


def _tr_usd(x):
    """USD degerini Turkce format ile gosterir ($1.234.567)."""
    try:
        return f"${float(x):,.0f}".replace(",", ".")
    except Exception:
        return str(x)


def fig_autofmt(fig):
    """Tarih eksenli grafiklerde x-etiketlerini okunur acida dondurur."""
    try:
        fig.autofmt_xdate(rotation=30)
    except Exception:
        pass


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

sim_months = st.sidebar.slider("Simulasyon Suresi (ay)", 1, 12, 6, step=1,
    help="Aylik tolerans (madde 1). Her ay 30 gun olarak hesaplanir; ornegin 6 ay = 180 gun. "
         "BOLUM B (makro analiz) ve grafik eksenleri secilen bu sureye gore olceklenir.")
days = int(sim_months * 30)
st.sidebar.caption(f"≈ {days} gun ({sim_months} ay) simule edilecek.")

st.sidebar.markdown("#### 🚗 Arac Sayisi (Filo)")
fleet_size = st.sidebar.slider("Kalici filo buyuklugu (arac)", 5, 150,
    30 if scenario == "FABRIKA" else 80,
    help="Sisteme kayitli, tum simulasyon boyunca tekrar tekrar sarj olan arac sayisi.")

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

with st.sidebar.expander("⚖️ Gerceklik Ayarlari: Fatura · Naive Modeli · Cikis Belirsizligi", expanded=False):
    billing_label = st.selectbox("Guc bedeli rejimi (N2)",
        ["EPDK (sozlesme gucu bazli — TR)", "Tepe bazli (demand charge — ABD tarzi)"],
        help="EPDK: guc bedeli SOZLESME GUCU uzerinden sabit tahakkuk eder; tepe dususu "
             "yalnizca ASIM CEZASINI azaltir + dusuk sozlesme firsati yaratir. "
             "Tepe bazli: olculen aylik 15-dk tepe x birim bedel (eski davranis).")
    billing_mode = "EPDK" if billing_label.startswith("EPDK") else "DEMAND"
    naive_label = st.selectbox("Algoritma-oncesi (naive) talep modeli (N1)",
        ["Cesitlilik tavani (talep tahmini)", "Olay-tabanli (tavansiz ust-sinir)"],
        help="Cesitlilik tavani: esZamanli talep = faktor x kurulu (VARSAYIM). "
             "Olay-tabanli: soketler taper'a gore serbest ceker; overload mumkun. "
             "Iki modu birlikte kosup BANT olarak raporlamak en durust sunumdur.")
    naive_mode = "diversity" if naive_label.startswith("Cesitlilik") else "event"
    dep_unc = st.slider("Cikis saati belirsizligi ±sigma (dk) (N7)", 0, 120, 0, 15,
        help="0 = optimizer cikis saatini TAM bilir (kesin beyan varsayimi). >0 ise "
             "planlama tahmini cikisla yapilir (gercek ± N(0, sigma)); tahmin hatasi "
             "tamamlanma oranini dusurebilir. Sahadaki gercege yakin test icin 30-60 dk deneyin.")

st.sidebar.markdown("#### 🤖 Algoritma Devreye Girme")
activation_label = st.sidebar.radio("Algoritma ne zaman calissin?",
    ["Her zaman", "Sadece puant (trafo doluluk ≥ %60)"],
    help="Puant: trafo doluluk orani (baz yuk/anma) ≥ %60. Disinda sistem "
         "algoritmasiz (yonetimsiz) calisir.")
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
           "PTF/SMF · IEC 60076-7 Termal Model · Gercek Ankara Sicakliklari · Resmi Tatiller (TR)")

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
            billing_mode=billing_mode, naive_mode=naive_mode, dep_unc=float(dep_unc),
        )
        st.session_state["payload"] = run_simulation(base_mult=1.0, **st.session_state["params"])
        # Baz yuk ARTIS senaryolari (+%10/+%15/+%20) her calistirmada OTOMATIK
        # uretilir (madde 2): asim cezasi + trafo omur tuketimi yeniden hesaplanir.
        st.session_state["PBASE"] = {
            pct: run_simulation(base_mult=1.0 + pct / 100.0, **st.session_state["params"])
            for pct in (10, 15, 20)
        }
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
gr[0].metric("Trafo Etkin Anma", f"{fmt_num(rated)} kW", f"{fmt_num(P['transformer_kva'])} kVA × {fmt_num(P['power_factor'], 2)}")
gr[1].metric("Kurulu İstasyon", f"{fmt_num(installed)} kW", f"cesitlilik {fmt_num(div, 2)}")
gr[2].metric("Cesitlilikli Talep", f"{fmt_num(diversified)} kW", f"%{fmt_num(div_pct)} trafo (hedef %20-30)")
gr[3].metric("Baz Yuk Tepe", f"{fmt_num(P['base_max_kw'])} kW", f"%{fmt_num(base_pct)} trafo (hedef %60)")

if naive_peak_pct <= 100.0:
    st.success(f"✅ **Overload YOK** (madde 9): algoritma ÖNCESI (algoritmasiz) tepe "
        f"**{fmt_num(P['peak_naive'])} kW** = trafo anmasinin **%{fmt_num(naive_peak_pct)}**'i ≤ %100. "
        f"Cesitlilik faktorlu istasyon talebi (%{fmt_num(div_pct)}) + baz tepe (%{fmt_num(base_pct)}) "
        f"trafoyu asmaz; DLM faydasi peak-shaving, maliyet, termal omur ve SOH'tur.")
else:
    st.warning(f"⚠️ Algoritma oncesi (algoritmasiz) tepe **{fmt_num(P['peak_naive'])} kW** trafo "
        f"anmasini (%{fmt_num(naive_peak_pct)}) asiyor. Istasyon sayisini azaltin veya cesitlilik "
        f"faktorunu dusurun (madde 9: algoritma oncesi bile overload olmamali).")
if not (20.0 <= div_pct <= 30.0):
    st.info(f"ℹ️ Cesitlilikli istasyon talebi trafo anmasinin **%{fmt_num(div_pct)}**'i "
        f"(hedef aralik %20-%30). Istasyon sayisini/cesitlilik faktorunu buna gore ayarlayin.")

st.caption(
    f"⚖️ Aktif gerceklik ayarlari: guc bedeli rejimi **{P['billing_mode']}** "
    f"({'sozlesme bazli sabit; tasarruf = ceza farki + dusuk sozlesme firsati' if P['billing_mode'] == 'EPDK' else 'olculen tepe bazli'}) · "
    f"tepe olcumu **15-dk ortalama** (EPDK/OSOS; naive {fmt_num(P['peak15_naive'])} / opt {fmt_num(P['peak15_opt'])} kW) · "
    f"naive modeli **{'cesitlilik tavani' if P['naive_mode'] == 'diversity' else 'olay-tabanli (ust-sinir)'}** · "
    f"cikis belirsizligi **±{P['dep_unc']:.0f} dk** · "
    f"SOH kazanci operator ROI'sine **{'dahil (filo tesisin mali)' if P['is_factory'] else 'dahil DEGIL (musteri bataryasi — ayri kalem)'}**.")

top = st.columns(4)
top[0].metric(f"Toplam Tasarruf ({P['cfg_days']} gun)", fmt_tl(A["total_saving_tl"]))
top[1].metric("Yillik Projeksiyon", fmt_tl(A["annual_total_saving_tl"]))
top[2].metric("Trafo Tepe (Once→Sonra)",
              f"{fmt_num(P['peak_naive'])}→{fmt_num(P['peak_opt'])} kW")
top[3].metric("Filo / Oturum", f"{P['fleet_size']} / {P['n_sessions']}")

# B1: Tamamlanma orani (algoritmanin araclari ac birakmadiginin kaniti) + ekstra KPI
comp_opt = A["costs_opt"]["completion_rate"] * 100.0
comp_naive = A["costs_naive"]["completion_rate"] * 100.0
k = st.columns(4)
k[0].metric("Tamamlanma — Algoritmali", f"%{fmt_num(comp_opt, 1)}",
            f"{fmt_num(comp_opt - comp_naive, 1, sign=True)} puan vs algoritmasiz",
            help="%80 SoC'ye ulasan oturum orani. Algoritma araclari ac BIRAKMAMALIDIR; "
                 "bu KPI optimizasyonun tamamlanmayi feda etmedigini kanitlar.")
k[1].metric("Tamamlanma — algoritmasiz", f"%{fmt_num(comp_naive, 1)}")
k[2].metric("Sozlesme Gucu", f"{fmt_num(P['contracted_kw'])} kW",
            help="Guc asim cezasi bu esigin uzerinde baslar. Optimize tepe bu degere cekilir.")
k[3].metric("Ort. Sarj Suresi (Algo)", f"{fmt_num(A['costs_opt']['avg_charge_duration_min'])} dk",
            f"algoritmasiz {fmt_num(A['costs_naive']['avg_charge_duration_min'])} dk",
            help="Ortalama tekil oturum sarj suresi. Madde 3 (guc tabani, S=3) ile "
                 "asiri uzamasi engellenir.")

sim_months_disp = P["cfg_days"] / 30.0
tabA, tabB = st.tabs(["📅 BOLUM A · 1 Gunluk Mikro Analiz",
                      f"📈 BOLUM B · {sim_months_disp:.0f} Aylik ({P['cfg_days']} gun) Makro Analiz"])

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

    with st.expander("🔋 Kombine Yuk ve Maliyet Egrisi (interaktif — imlecle her saatin kW/fiyatini oku)", expanded=True):
        st.caption("Madde 4: Grafik INTERAKTIFTIR. Fareyi egri uzerinde herhangi bir saate (t) "
                   "goturdugunde o andaki TUM kW degerleri ve fiyat (PTF/tarife) birlikte gosterilir. "
                   "Yakinlastirma/kaydirma ve seri ac-kapa (lejant tiklamasi) desteklenir.")
        g = st.columns(6)
        show_base = g[0].checkbox("Baz Yuk", True)
        show_naive = g[1].checkbox("Algoritma Oncesi", True)
        show_opt = g[2].checkbox("Algoritma Sonrasi", True)
        show_rated = g[3].checkbox("Trafo Anma", True)
        show_contract = g[4].checkbox("Sozlesme Gucu", True)
        show_price = g[5].checkbox("Fiyat Egrisi", True)

        plabel = "PTF (TL/kWh)" if P["pricing_mode"] == "PTF" else "Tarife (TL/kWh)"
        figp = go.Figure()
        if show_base:
            figp.add_trace(go.Scatter(x=x, y=base, name="Baz Yuk", fill="tozeroy",
                line=dict(color="#5f6368", width=1.5), hovertemplate="Baz: %{y:.0f} kW<extra></extra>"))
        if show_naive:
            figp.add_trace(go.Scatter(x=x, y=fac_naive, name="Algoritma Oncesi (algoritmasiz)",
                line=dict(color="#d93025", width=2.4),
                hovertemplate="Oncesi: %{y:.0f} kW<extra></extra>"))
        if show_opt:
            figp.add_trace(go.Scatter(x=x, y=fac_opt, name="Algoritma Sonrasi (optimize)",
                line=dict(color="#1e8e3e", width=2.4),
                hovertemplate="Sonrasi: %{y:.0f} kW<extra></extra>"))
        if show_rated:
            figp.add_hline(y=P["rated_kw"], line=dict(color="black", dash="dash", width=1.2),
                annotation_text=f"Trafo Anma {fmt_num(P['rated_kw'])} kW", annotation_position="top left")
        if show_contract:
            figp.add_hline(y=P["contracted_kw"], line=dict(color="#ea8600", dash="dashdot", width=1.2),
                annotation_text=f"Sozlesme {fmt_num(P['contracted_kw'])} kW", annotation_position="bottom left")
        if show_price:
            figp.add_trace(go.Scatter(x=x, y=price_d, name=plabel, yaxis="y2",
                line=dict(color="#1a73e8", width=1.8, dash="dot"),
                hovertemplate="Fiyat: %{y:.3f} TL/kWh<extra></extra>"))
        figp.update_layout(
            height=460, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Saat", dtick=2, range=[0, 24],
                       hoverformat=".2f", ticksuffix=":00"),
            yaxis=dict(title="Guc (kW)"),
            yaxis2=dict(title=plabel, overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(figp, use_container_width=True)
        # Ek olarak: secilen saatte tam okuma (slider) — grafik hover'ina alternatif kesin deger.
        t_hour = st.slider("İmleç — saat (t) sec", 0.0, 23.98, 12.0, 0.25, key="cursor_hour",
                           help="Grafikteki herhangi bir t anini bu imlecle de secip kesin kW/fiyat okuyabilirsiniz.")
        ti = int(round(t_hour * 60)) % MINUTES_PER_DAY
        rc = st.columns(4)
        rc[0].metric("Baz Yuk @t", f"{fmt_num(base[ti])} kW")
        rc[1].metric("Oncesi (algoritmasiz) @t", f"{fmt_num(fac_naive[ti])} kW")
        rc[2].metric("Sonrasi (optimize) @t", f"{fmt_num(fac_opt[ti])} kW",
                     f"{fmt_num(fac_opt[ti]-fac_naive[ti], sign=True)} kW")
        rc[3].metric(f"Fiyat @t ({'PTF' if P['pricing_mode']=='PTF' else 'Tarife'})",
                     f"{fmt_num(price_d[ti], 3)} TL/kWh")

    with st.expander("⚡ Sarj Egrisi · Algoritma Oncesi vs Sonrasi (madde 8)", expanded=True):
        st.caption("Yalnizca SARJ gucu (baz yuk haric). Algoritma oncesi (algoritmasiz) "
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
                   "aracin algoritma ONCESI (algoritmasiz) ve SONRASI (optimize) sarj suresi. "
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
            dcol[1].metric("Ort. sure — Oncesi", f"{fmt_num(np.nanmean(day_tbl['charge_duration_min_naive']))} dk")
            dcol[2].metric("Ort. sure — Sonrasi", f"{fmt_num(np.nanmean(day_tbl['charge_duration_min_opt']))} dk",
                           f"{fmt_num(np.nanmean(day_tbl['Uzama (dk)']), sign=True)} dk ort. uzama")
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
    cc[3].metric("Tasarruf (%)", f"%{fmt_num(saving_pct, 1)}")

    shave, pct, extra, ev_inc = day_roi(P, float(fac_opt.max()), float(fac_naive.max()))
    st.markdown("#### 🏭 Rezerv Yuk (Power Shaving) Yatirim Getirisi")
    st.success(
        f"Trafo tepe yukunde **%{fmt_num(pct)}** ({fmt_num(shave)} kW) tirasama yapildi; ayni trafoda "
        f"**{fmt_num(shave)} kW** buyume payi (headroom) olustu. Yeni talebin de DLM tavani "
        f"altinda yonetilmesi KOSULUYLA ilave **{extra} adet** DC istasyona alan acilir; "
        f"desteklenen EV kapasitesi **%{fmt_num(ev_inc)}** artirilabilir. Trafo-tarafi parasal "
        f"kazanc, asim cezalarinin onlenmesi ve sozlesme gucunun dusuk tutulabilmesidir "
        f"(bu boyutlandirmada termal omur kazanci kucuktur — bkz. Bolum B termal panel).")

    # ---- BOŞTA KALAN TRAFO GÜCÜ (HEADROOM) — bu gun (t), Oncesi vs Sonrasi ----
    contract = P["contracted_kw"]
    peak_n_day = float(fac_naive.max()); peak_o_day = float(fac_opt.max())
    idle_n = rated - peak_n_day; idle_o = rated - peak_o_day          # tepe anindaki bosta guc
    idle_n_mean = rated - float(fac_naive.mean())
    idle_o_mean = rated - float(fac_opt.mean())                       # gun-ortalamasi bosta guc
    with st.expander("🅿️ Bosta Kalan Trafo Gucu (Headroom) — Bu Gun · Oncesi vs Sonrasi", expanded=True):
        st.caption(
            f"{day+1}. gun ({sel_date.strftime('%d.%m.%Y')}) icin TEPE anindaki bosta "
            "(kullanilmayan) trafo gucu = trafo etkin anma − tepe yuk. Bu deger, kurulu "
            "trafo gucune ve sozlesme gucune oranlanir. Bosta kapasite hem 'bedeli odenip "
            "kullanilmayan' bir kayip hem de ilave EV/istasyon icin bir firsat (rezerv) anlamina gelir.")
        hc = st.columns(2)
        with hc[0]:
            st.markdown("**🔴 Algoritma Oncesi (algoritmasiz)**")
            st.metric("Tepe Yuk", f"{fmt_num(peak_n_day)} kW", f"%{fmt_num(peak_n_day/rated*100)} trafo doluluk")
            st.metric("Bosta Trafo Gucu (tepe ani)", f"{fmt_num(idle_n)} kW",
                      f"gun ort. {fmt_num(idle_n_mean)} kW", delta_color="off")
            st.metric("↳ Trafo kurulu gucune oran", f"%{fmt_num(idle_n/rated*100)}",
                      help="Bosta guc / trafo etkin anma (1520 kW).")
            st.metric("↳ Sozlesme gucune oran", f"%{fmt_num(idle_n/contract*100)}",
                      help=f"Bosta guc / sozlesme gucu ({fmt_num(contract)} kW).")
        with hc[1]:
            st.markdown("**🟢 Algoritma Sonrasi (optimize)**")
            st.metric("Tepe Yuk", f"{fmt_num(peak_o_day)} kW", f"{fmt_num(peak_o_day-peak_n_day, sign=True)} kW vs oncesi")
            st.metric("Bosta Trafo Gucu (tepe ani)", f"{fmt_num(idle_o)} kW",
                      f"{fmt_num(idle_o-idle_n, sign=True)} kW vs oncesi")
            st.metric("↳ Trafo kurulu gucune oran", f"%{fmt_num(idle_o/rated*100)}",
                      f"{fmt_num((idle_o-idle_n)/rated*100, sign=True)} puan")
            st.metric("↳ Sozlesme gucune oran", f"%{fmt_num(idle_o/contract*100)}",
                      f"{fmt_num((idle_o-idle_n)/contract*100, sign=True)} puan")
        st.info(
            f"💡 **Finansal anlam:** Tepe aninda trafonun **%{fmt_num(idle_o/rated*100)}**'i (≈{fmt_num(idle_o)} kW) "
            f"bostadir. Algoritma tepeyi tirasayarak bosta gucu **{fmt_num(idle_n)}→{fmt_num(idle_o)} kW** "
            f"(+{fmt_num(idle_o-idle_n)} kW) buyuttu. Bu rezerv ya **ilave EV/istasyon geliri**ne donusturulur, "
            f"ya da **daha dusuk sozlesme gucu** secilerek aylik guc bedeli dusurulur (asagidaki duyarlilik). "
            f"Sozlesme gucune gore bosta pay %{fmt_num(idle_n/contract*100)}→%{fmt_num(idle_o/contract*100)}; "
            f"yani sozlesme esigi ile tepe arasinda {fmt_num(contract-peak_o_day, sign=True)} kW guvenli marj kalir "
            f"({'asim/ceza yok' if peak_o_day <= contract else 'tepe sozlesmeyi asiyor — ceza riski'}).")

        st.caption("ℹ️ Sozlesme gucu artirmanin maliyeti ve uzun-vadeli finansal karsiliklar "
                   "BOLUM B'de (makro analiz) detaylandirilir.")

# =========================================================================== #
# BOLUM B
# =========================================================================== #
with tabB:
    th_o = A["thermal_opt"]; th_n = A["thermal_naive"]; soh = A["soh"]
    days_axis = np.arange(P["cfg_days"]) + 1
    # TAKVIM EKSENI (madde 1): grafikler gun yerine GERCEK TARIH ile etiketlenir;
    # boylece secilen sure (ornegin 6 ay) eksende dogrudan ay olarak gorunur.
    date_axis = [START_DATE + _dt.timedelta(days=int(i)) for i in range(P["cfg_days"])]
    period_months = P["cfg_days"] / 30.0
    period_lbl = f"{period_months:.0f} ay ({P['cfg_days']} gun)"
    annual_factor_b = 365.0 / max(P["cfg_days"], 1)
    usd_rate = P["usd_try_rate"]; tcost_tl = P["transformer_cost_tl"]
    design_life = P["design_life_years"]

    lp = A["life_proj"]; pj_o = lp["proj_opt"]; pj_n = lp["proj_naive"]
    with st.expander("🔥 Trafo Omru (30 Yil Bazli) - IEC 60076-7 · Oncesi vs Sonrasi (madde 2)", expanded=True):
        st.caption(f"Trafo NORMAL/TASARIM omru **{fmt_num(design_life)} yil** alinir (referans yaslanma "
                   f"hizinda, sicak-nokta 98°C). Sicak-nokta IEC 60076-7:2018 fark-denklemi + GERCEK "
                   f"Ankara ortam sicakligiyla (tepe ~{fmt_num(A['ambient_peak_c'])}°C) hesaplanir; bagil "
                   f"yaslanma V=2^((θh−98)/6). Asagida, ÖNCE/SONRA omur tuketimi {fmt_num(design_life)} yila "
                   f"oranla yuzdesel verilir ve fark trafo maliyetiyle parasallastirilir.")
        # Bu DONEMDE (sim penceresi) tuketilen omur, 30 yillik butceye (262.800 saat) oranla %
        pct_period_n = th_n["pct_life_consumed"]      # = lol_hours / (30y*8760) * 100
        pct_period_o = th_o["pct_life_consumed"]
        # 30 yil boyunca bu yuk profili surerse tuketilecek omur (mevsim duzeltmeli projeksiyon)
        pct_30y_n = pj_n["pct_life_horizon"]; pct_30y_o = pj_o["pct_life_horizon"]
        save_period_tl = A["thermal_saving_tl"]                  # (Δ donem tuketimi) × trafo maliyeti
        save_30y_tl = A.get("thermal_saving_deferred_tl", 0.0)   # 30y projeksiyon farki × maliyet

        k1, k2, k3 = st.columns(3)
        k1.metric("Tepe Sicak-Nokta (Once→Sonra)",
                  f"{fmt_num(th_n['theta_hs_peak'])}→{fmt_num(th_o['theta_hs_peak'])} °C",
                  help="98°C altinda yaslanma referanstan (normalden) yavastir.")
        k2.metric(f"Bu Donemde ({period_lbl}) Tuketilen Omur",
                  f"%{fmt_num(pct_period_n, 3)}→%{fmt_num(pct_period_o, 3)}",
                  f"{fmt_num(pct_period_o-pct_period_n, 3, sign=True)} puan",
                  help=f"Bu simulasyon doneminde tuketilen omur, {fmt_num(design_life)} yillik toplam "
                       f"omre oranla. (Δ × trafo maliyeti = donem trafo tasarrufu.)")
        k3.metric("30 Yilda Tuketilen Omur (projeksiyon)",
                  f"%{fmt_num(pct_30y_n, 2)}→%{fmt_num(pct_30y_o, 2)}",
                  help=f"Bu yuk profili 30 yil surerse (mevsim faktoru {fmt_num(lp['seasonal_factor'], 2)}) "
                       f"tuketilecek omur yuzdesi. %100 = 30 yil sonunda omur biter.")

        st.success(
            f"💰 **Algoritmanin trafo omrune kazanci (madde 2):** Bu donemde "
            f"({period_lbl}) omur tuketimi **%{fmt_num(pct_period_n, 3)} → %{fmt_num(pct_period_o, 3)}** "
            f"({fmt_num(pct_period_n-pct_period_o, 3, sign=True)} puan dusus). Bu farkin {fmt_num(design_life)} yillik "
            f"omur butcesine ({fmt_tl_usd(tcost_tl, usd_rate)} trafo) parasal karsiligi: "
            f"**{fmt_tl_usd(save_period_tl, usd_rate)}** (bu donem). Yillik projeksiyon: "
            f"**{fmt_tl_usd(save_period_tl*annual_factor_b, usd_rate)}/yil**; "
            f"30 yil profili surdugunde korunan omur: **{fmt_tl_usd(save_30y_tl, usd_rate)}**.")
        st.caption(f"Trafo yenileme maliyeti: **{fmt_tl_usd(tcost_tl, usd_rate)}** "
                   f"(USD/TRY={fmt_num(usd_rate, 1)}). Not: Dogru boyutlandirilmis (overload'suz) bu sistemde "
                   f"termal yaslanma dusuktur; trafo omru cogunlukla 30 yillik FIZIKSEL tasarim omruyle "
                   f"sinirlidir. Asil buyuk trafo-tarafi kazanc, asim/overload CEZALARININ onlenmesidir "
                   f"(asagidaki bolumler).")

        figt, axt = plt.subplots(figsize=(12, 3.8))
        # Kumulatif yaslanmayi 30-yil omur YUZDESI olarak goster (saat yerine anlamli %)
        life_hours_budget = design_life * 8760.0
        cum_n_pct = th_n["cum_aging_hours_daily"] / life_hours_budget * 100.0
        cum_o_pct = th_o["cum_aging_hours_daily"] / life_hours_budget * 100.0
        axt.plot(date_axis, cum_n_pct, color="#d93025", lw=2, label="Algoritmasiz")
        axt.plot(date_axis, cum_o_pct, color="#1e8e3e", lw=2, label="Algoritmali")
        axt.fill_between(date_axis, cum_o_pct, cum_n_pct, color="#fbbc04", alpha=0.25,
                         label="Onlenen Omur Tuketimi")
        axt.set_xlabel("Tarih"); axt.set_ylabel(f"Tuketilen Omur (%/{fmt_num(design_life)} yil)")
        axt.grid(alpha=0.25); axt.legend(fontsize=9); fig_autofmt(figt)
        figt.tight_layout(); st.pyplot(figt); plt.close(figt)

    with st.expander("🔋 Batarya Sagligi (SOH) Kumulatif Analizi", expanded=True):
        s1, s2, s3 = st.columns(3)
        s1.metric("Ort. SOH Dususu — Algoritmasiz", f"%{fmt_num(soh['final_soh_drop_naive_pct'], 3)}")
        s2.metric("Ort. SOH Dususu — Algoritmali", f"%{fmt_num(soh['final_soh_drop_opt_pct'], 3)}")
        s3.metric("Geciktirilen Batarya Degisim Bedeli", fmt_tl(soh["delayed_replacement_value_tl"]))
        figs, axs = plt.subplots(figsize=(12, 3.8))
        axs.plot(date_axis, soh["soh_naive_ts"], color="#d93025", lw=2, label="Algoritmasiz")
        axs.plot(date_axis, soh["soh_opt_ts"], color="#1e8e3e", lw=2, label="Algoritmali")
        axs.fill_between(date_axis, soh["soh_opt_ts"], soh["soh_naive_ts"],
                         color="#fbbc04", alpha=0.25, label="Korunan SOH")
        axs.set_xlabel("Tarih"); axs.set_ylabel("Filo Ortalama SOH (%)")
        axs.grid(alpha=0.25); axs.legend(fontsize=9); fig_autofmt(figs)
        figs.tight_layout(); st.pyplot(figs); plt.close(figs)

    with st.expander("🚗 Arac Bazli Mikro Karsilastirma Tablosu", expanded=True):
        st.caption("Her arac icin SOH dususu, toplam sarj suresi, EN KÖTÜ tekil oturumun "
                   "MUTLAK uzamasi (dk) ve TOPLAM surenin yuzde uzamasi (Σopt/Σnaive). "
                   "Yuzde, toplam-bazlidir (tekil-oturum yuzdesi hizli passenger araclarda "
                   "yaniltici biçimde sisiyordu).")
        st.dataframe(
            soh["table"].style.format({
                "SOH Dususu Algoritmali (%)": _tr(".4f"), "SOH Dususu Algoritmasiz (%)": _tr(".4f"),
                "SOH Korunan (puan)": _tr(".4f"),
                "Toplam Sarj Suresi Algoritmali (dk)": _tr(",.0f"),
                "Toplam Sarj Suresi Algoritmasiz (dk)": _tr(",.0f"),
                "Maks Sarj Uzatma (dk)": _tr(".0f"), "Toplam Sure Uzamasi (%)": _tr(".1f"),
                "Korunan Batarya Bedeli (TL)": _tr(",.0f"),
            }), use_container_width=True, hide_index=True, height=430)

    # ---- Senaryo: Baz Yuk Artisi +%10/+%15/+%20 (her calistirmada otomatik) ----
    with st.expander("📊 Baz Yuk Artis Senaryolari (+%10 / +%15 / +%20) → Ceza + Trafo Omru", expanded=True):
        st.caption("Ayni istasyon ve araclar sabit; baz yuk %10-%20 artirildiginda algoritma "
                   "ONCESI ve SONRASI icin guc asim cezasi (EPDK, 15-dk tepe) ve TRAFO OMUR "
                   "TUKETIMI yeniden hesaplanir. Sicak-nokta arttikca yaslanma USTEL (V=2^((θh−98)/6)) "
                   "buyudugu icin, baz yuk buyudukce algoritmanin TERMAL kazanci da buyur.")
        if "PBASE" in st.session_state:
            scen_list = [("Mevcut (+%0)", P)] + [
                (f"+%{pct}", st.session_state["PBASE"][pct]) for pct in sorted(st.session_state["PBASE"])
            ]
            rows_b = []
            for nm, R in scen_list:
                AR = R["analysis"]
                tn_r = AR["thermal_naive"]; to_r = AR["thermal_opt"]
                rows_b.append({
                    "Senaryo": nm,
                    "Baz Tepe (kW)": R["base_max_kw"],
                    "15-dk Tepe Oncesi (kW)": R["peak15_naive"],
                    "15-dk Tepe Sonrasi (kW)": R["peak15_opt"],
                    "Asim Cezasi Oncesi (TL)": R["demand_penalty_naive"],
                    "Asim Cezasi Sonrasi (TL)": R["demand_penalty_opt"],
                    "θhs Tepe Once→Sonra (°C)": f"{fmt_num(tn_r['theta_hs_peak'])}→{fmt_num(to_r['theta_hs_peak'])}",
                    "Omur Tuketimi Oncesi (%)": tn_r["pct_life_consumed"],
                    "Omur Tuketimi Sonrasi (%)": to_r["pct_life_consumed"],
                    "Termal Tasarruf (TL)": AR["thermal_saving_tl"],
                    "Toplam Tasarruf (TL)": AR["total_saving_tl"],
                })
            df_b = pd.DataFrame(rows_b)
            st.dataframe(df_b.style.format({
                "Baz Tepe (kW)": _tr(".0f"),
                "15-dk Tepe Oncesi (kW)": _tr(".0f"), "15-dk Tepe Sonrasi (kW)": _tr(".0f"),
                "Asim Cezasi Oncesi (TL)": _tr(",.0f"), "Asim Cezasi Sonrasi (TL)": _tr(",.0f"),
                "Omur Tuketimi Oncesi (%)": _tr(".4f"), "Omur Tuketimi Sonrasi (%)": _tr(".4f"),
                "Termal Tasarruf (TL)": _tr(",.0f"), "Toplam Tasarruf (TL)": _tr(",.0f"),
            }), use_container_width=True, hide_index=True)

            P20 = st.session_state["PBASE"][max(st.session_state["PBASE"])]
            A20 = P20["analysis"]
            th_gain_now = A["thermal_saving_tl"]
            th_gain_20 = A20["thermal_saving_tl"]
            pen_gain_20 = P20["demand_penalty_naive"] - P20["demand_penalty_opt"]
            st.error(
                f"Baz yuk +%20 oldugunda **algoritmasiz** 15-dk tepe {fmt_num(P20['peak15_naive'])} kW'a "
                f"cikar; asim cezasi **{fmt_tl(P20['demand_penalty_naive'])}** olur. **Algoritma** "
                f"tepeyi {fmt_num(P20['peak15_opt'])} kW'a cekerek cezayi "
                f"**{fmt_tl(P20['demand_penalty_opt'])}**'ye indirir (onlenen: {fmt_tl(pen_gain_20)}).")
            st.success(
                f"🔥 **Termal kazanc baz yukle birlikte buyur:** donem termal tasarrufu "
                f"mevcutta {fmt_tl(th_gain_now)} iken +%20 baz yukte "
                f"**{fmt_tl(th_gain_20)}**'ye cikar (×{fmt_num(th_gain_20/max(th_gain_now,1e-9), 1)}). "
                f"Sicak-nokta yukseldikce yaslanma ustel hizlandigi icin, trafo dolulugu "
                f"arttikca DLM'in trafo-omru getirisi de buyur — tesis buyudukce algoritmanin "
                f"degeri artar.")
            st.info("Ceza = Σ_ay max(0, 15-dk ortalama tepe − sozlesme) × güç bedeli × ceza katı "
                    "(EPDK/OSOS — N3). Omur tuketimi: IEC 60076-7 fark-denklemi, donem yuzdesi "
                    "30 yillik omur butcesine gore.")

    # ---- FILO BUYUME KAPASITESI: cezasiz maks arac sayisi (madde 3) ----
    with st.expander("🚛 Filo Buyume Kapasitesi — Cezasiz Maks Arac Sayisi (Oncesi vs Sonrasi)", expanded=True):
        st.caption("Mevcut soket kurulumu ve sozlesme gucu SABITKEN: algoritma ONCESI (algoritmasiz) "
                   "kac araclik filo, trafo/sozlesme ASIM CEZASI yemeden sarj edilebilir? Ayni soru "
                   "algoritma SONRASI icin de cevaplanir (optimize tepeyi sozlesmeye kirptigi icin "
                   "ceza olusmaz; sinir, hizmet kalitesidir: tamamlanma ≥ %98). Arama hizi icin "
                   "30 gunluk pencere kullanilir; sonuc ± birkac arac belirsizlik tasir (stokastik oturumlar).")
        if st.button("🚛 Maks Filoyu Hesapla"):
            SEARCH_DAYS = 30
            FLEET_HI = 300

            def _fleet_run(n_veh):
                bp = dict(st.session_state["params"])
                bp["fleet_size"] = int(n_veh); bp["days"] = SEARCH_DAYS
                return run_simulation(base_mult=1.0, **bp)

            def _naive_ok(r):       # cezasiz: 15-dk tepe sozlesmeyi asmiyor
                return r["demand_penalty_naive"] <= 0.5

            def _opt_ok(r):         # ceza zaten ~0; sinir hizmet kalitesi
                return (r["demand_penalty_opt"] <= 0.5
                        and r["analysis"]["costs_opt"]["completion_rate"] >= 0.98)

            def _max_fleet(ok_fn, prog, p0, p1):
                lo, hi = 2, FLEET_HI
                if not ok_fn(_fleet_run(lo)):
                    return 0
                prog.progress(p0 + 0.1 * (p1 - p0))
                if ok_fn(_fleet_run(hi)):
                    return hi
                step = 0
                while hi - lo > 1:                 # ikili arama (~8 adim)
                    mid = (lo + hi) // 2
                    if ok_fn(_fleet_run(mid)):
                        lo = mid
                    else:
                        hi = mid
                    step += 1
                    prog.progress(min(p0 + (0.1 + 0.9 * step / 9.0) * (p1 - p0), p1))
                return lo

            prog = st.progress(0.0, text="Algoritma oncesi (cezasiz) maks filo araniyor...")
            n_naive = _max_fleet(_naive_ok, prog, 0.0, 0.5)
            prog.progress(0.5, text="Algoritma sonrasi (tamamlanma ≥ %98) maks filo araniyor...")
            n_opt = _max_fleet(_opt_ok, prog, 0.5, 1.0)
            prog.empty()
            st.session_state["fleet_cap"] = (n_naive, n_opt)
        if "fleet_cap" in st.session_state:
            n_naive, n_opt = st.session_state["fleet_cap"]
            growth = n_opt - n_naive
            growth_pct = (growth / max(n_naive, 1)) * 100.0
            fc = st.columns(3)
            fc[0].metric("Maks Filo — Algoritma Oncesi", f"{n_naive} arac",
                         help="Bu sayinin ustunde algoritmasiz isletme, sozlesme gucunu (15-dk tepe) "
                              "asip EPDK asim cezasi yemeye baslar.")
            fc[1].metric("Maks Filo — Algoritma Sonrasi",
                         f"{n_opt} arac" + (" (ust sinir)" if n_opt >= 300 else ""),
                         f"+{growth} arac vs oncesi",
                         help="Optimize tepe sozlesmeye kirpildigi icin ceza HIC olusmaz; sinir, "
                              "tamamlanma oraninin %98 altina dusmesidir (soket sayisi belirleyici olabilir).")
            fc[2].metric("Filo Buyume Kapasitesi", f"%{fmt_num(growth_pct)}",
                         help="Ayni trafo, ayni soketler, ayni sozlesme gucuyle desteklenebilen ek filo.")
            if n_opt >= 300:
                st.info("ℹ️ Algoritma sonrasi limit arama tavanina (300) dayandi; gercek sinir soket "
                        "sayisi/kuyruk dinamigiyle belirlenir. Soket ekleyerek tekrar deneyin.")
            st.success(
                f"💼 **Yatirimci mesaji:** Ayni trafo ve sozlesme gucuyle, algoritmasiz isletme "
                f"**{n_naive} araclik** filoda ceza sinirina dayanir. DLM ile filo, ceza riski "
                f"OLMADAN ve %98+ tamamlanma korunarak **{n_opt} araca** buyutulebilir — "
                f"**+{growth} arac (%{fmt_num(growth_pct)} buyume)**, ilave trafo/sozlesme yatirimi "
                f"gerektirmeden.")

    # ---- SOZLESME GUCU ARTIRMA MALIYETI (madde 4, 5) ----
    dc_b = P["demand_charge_tl_per_kw"]
    contract_b = P["contracted_kw"]
    peak_nb = P["peak_naive"]; peak_ob = P["peak_opt"]
    energy_y_ob = A["costs_opt"]["energy_cost_tl"] * annual_factor_b
    gucbedeli_y_b = contract_b * dc_b * 12.0                       # yillik guc bedeli (sozlesme bazli)
    pen_y_ob = P["demand_penalty_opt"] * annual_factor_b
    total_y_ob = energy_y_ob + gucbedeli_y_b + pen_y_ob           # yillik toplam fatura (opt)
    with st.expander("📑 Sozlesme Gucu Artirma Maliyeti — %X artis → %Y maliyet (madde 4)", expanded=True):
        st.caption(
            "Sozlesme gucu, guc bedelinin (demand charge) tahakkuk tabanidir (EPDK: yillik guc "
            "bedeli ≈ sozlesme gucu × birim bedel × 12 ay) ve tepe yuk bu esigi asarsa asim cezasi "
            "baslar. Sozlesme gucunu artirmanin yillik maliyete (hem TL hem USD) etkisi:")
        rows = []
        for xp in (10, 20, 30):
            newC = contract_b * (1 + xp / 100.0)
            extra = (xp / 100.0) * contract_b * dc_b * 12.0
            rows.append({
                "Artis": f"+%{xp}",
                "Yeni Sozlesme (kW)": newC,
                "Yillik Guc Bedeli (TL)": newC * dc_b * 12.0,
                "Ek Maliyet/yil (TL)": extra,
                "Ek Maliyet/yil (USD)": extra / usd_rate,
                "Toplam Faturaya Etki": f"%{fmt_num(extra/total_y_ob*100, 1)}",
            })
        st.dataframe(pd.DataFrame(rows).style.format({
            "Yeni Sozlesme (kW)": _tr(".0f"), "Yillik Guc Bedeli (TL)": _tr(",.0f"),
            "Ek Maliyet/yil (TL)": _tr(",.0f"), "Ek Maliyet/yil (USD)": _tr_usd,
        }), use_container_width=True, hide_index=True)
        st.warning(
            f"📈 Ornek: Sozlesme gucunu **%20** artirmak ({fmt_num(contract_b)}→{fmt_num(contract_b*1.2)} kW), "
            f"yillik guc bedelini **{fmt_tl_usd(0.2*contract_b*dc_b*12, usd_rate)}** = toplam faturanin "
            f"**%{fmt_num(0.2*contract_b*dc_b*12/total_y_ob*100, 1)}**'i kadar artirir (enerji maliyeti degismez).")
        red_kw = max(0.0, peak_nb - peak_ob)
        red_pct = (red_kw / peak_nb * 100.0) if peak_nb > 0 else 0.0
        save_y_contract = red_kw * dc_b * 12.0
        st.success(
            f"✅ **Algoritmanin getirisi (ters yon):** Tepe yuk {fmt_num(peak_nb)}→{fmt_num(peak_ob)} kW "
            f"(−%{fmt_num(red_pct)}) dustugu icin, ayni operasyonu **{fmt_num(red_kw)} kW daha dusuk** sozlesme "
            f"gucuyle (asim cezasiz) yurutebilirsiniz → yillik **{fmt_tl_usd(save_y_contract, usd_rate)}** "
            f"guc bedeli tasarrufu. DLM, sozlesme gucu artirmaya gerek BIRAKMADAN buyumeyi mumkun kilar.")

    # ---- UZUN VADELI FINANSAL OZET (madde 5): tum kalemler tek tabloda ----
    with st.expander("💼 Uzun Vadeli Finansal Ozet — Tum Kazanim/Maliyet Kalemleri (madde 5)", expanded=True):
        st.caption(f"Finansal karsiligi olan tum uzun-vadeli kalemler. Donem = {period_lbl}. "
                   f"Yillik projeksiyon MEVSIM DUZELTMELIDIR (N4): enerji ×{fmt_num(annual_factor_b, 2)}"
                   f"×{fmt_num(A['energy_seasonal_factor'], 2)} (solar makasi kisin sig), termal "
                   f"×{fmt_num(annual_factor_b, 2)}×{fmt_num(A['life_proj']['seasonal_factor'], 2)} (kis ortami serin); "
                   f"demand/SOH lineer. Guc bedeli rejimi: **{P['billing_mode']}**.")
        items = [
            ("Toplam Enerji Tasarrufu", A["energy_saving_tl"], A["annual_energy_saving_tl"],
             f"Ucuz saatlere kaydirma (PTF/tarife) ile dogrudan OPEX dususu. "
             f"Yillik = ×{fmt_num(annual_factor_b, 2)}×{fmt_num(A['energy_seasonal_factor'], 2)} mevsim duzeltmesi."),
            ("Demand/Asim Tasarrufu", A["demand_saving_tl"], A["annual_demand_saving_tl"],
             ("EPDK rejimi: guc bedeli sozlesme bazli SABIT -> tasarruf yalnizca ASIM CEZASI "
              "farkidir (15-dk tepe olcumu)." if P["billing_mode"] == "EPDK"
              else "Tepe-bazli rejim: aylik 15-dk tepe × birim bedel + asim cezasi dususu.")),
            ("Guc Asim Cezasi (onlenen, alt-kalem)",
             P["demand_penalty_naive"] - P["demand_penalty_opt"],
             (P["demand_penalty_naive"] - P["demand_penalty_opt"]) * annual_factor_b,
             "Sozlesme asimi cezasi (EPDK ×3, 15-dk ortalama tepe); ustteki kalemin alt-detayi."),
            ("Trafo Omru Kazanci (30y bazli)", A["thermal_saving_tl"], A["annual_thermal_saving_tl"],
             f"(Δ omur tuketimi %) × trafo maliyeti ({fmt_num(design_life)} yil bazli). Dogru "
             f"boyutlandirilmis sistemde KUCUKTUR; asil trafo-tarafi kazanc asim/sozlesme tarafidir."),
        ]
        if A["soh_in_operator_roi"]:
            items.append(
                ("SOH (Batarya) Kazanci — filo tesisin mali", A["soh_saving_tl"], A["annual_soh_saving_tl"],
                 "Dusuk C-rate -> korunan batarya sagligi -> ertelenen batarya degisimi. "
                 "GOSTERGE niteligindedir (k=0.6 stres katsayisi kalibre edilmemistir)."))
        df_long = pd.DataFrame([{
            "Kalem": nm,
            f"Donem ({period_lbl}) TL": dv,
            "Yillik TL": yv,
            "Yillik USD": yv / usd_rate,
            "Aciklama": desc,
        } for (nm, dv, yv, desc) in items])
        st.dataframe(df_long.style.format({
            f"Donem ({period_lbl}) TL": _tr(",.0f"), "Yillik TL": _tr(",.0f"), "Yillik USD": _tr_usd,
        }), use_container_width=True, hide_index=True)
        if not A["soh_in_operator_roi"]:
            st.info(f"🔋 **SOH kazanci ({fmt_tl(A['soh_saving_tl'])}/donem) operator ROI'sine "
                    f"DAHIL EDILMEDI (N5):** AVM senaryosunda bataryalar MUSTERININ malidir. Bu "
                    f"deger, musteri sadakati/pazarlama arguman olarak ayrica kullanilabilir "
                    f"(gosterge niteliginde; SOH modeli kalibre edilmemistir).")
        tot_period = A["total_saving_tl"]; tot_annual = A["annual_total_saving_tl"]
        m_lt = st.columns(3)
        m_lt[0].metric(f"TOPLAM ({period_lbl})", fmt_tl(tot_period))
        m_lt[1].metric("TOPLAM (yillik, mevsim duzeltmeli)", fmt_tl(tot_annual))
        m_lt[2].metric("TOPLAM (yillik, USD)", fmt_usd(tot_annual / usd_rate))
        st.info("Ayrica sozlesme gucu artirma maliyeti yukaridaki '📑 Sozlesme Gucu Artirma' "
                "bolumunde; bu, algoritmanin ONLEDIGI bir gider kalemidir (tepe dusunce sozlesme "
                "artirmaya gerek kalmaz). EPDK rejiminde 'dusuk sozlesme firsati' oradaki yesil "
                "kutuda parasallastirilir.")

    st.markdown("#### 🧾 Makro Finansal Ozet")
    f = st.columns(4)
    f[0].metric("Enerji Tasarrufu", fmt_tl(A["energy_saving_tl"]))
    f[1].metric("Demand Charge Tasarrufu", fmt_tl(A["demand_saving_tl"]))
    f[2].metric("Trafo Omru Kazanci", fmt_tl(A["thermal_saving_tl"]),
                help=f"(Δ omur tuketimi %) × trafo maliyeti, {fmt_num(design_life)} yil bazli (madde 2). "
                     f"USD: {fmt_usd(A['thermal_saving_tl']/usd_rate)}.")
    f[3].metric("SOH (Batarya) Tasarrufu", fmt_tl(A["soh_saving_tl"]),
                help="GOSTERGE niteligindedir (k=0.6 kalibre edilmemis). AVM senaryosunda "
                     "musteri faydasidir; operator ROI toplamina dahil edilmez (N5).")

    # ---- DASHBOARD VERILERININ FINANSAL ANLAMI: Oncesi vs Sonrasi kiyasi ----
    with st.expander("🧭 Dashboard Verilerinin Finansal Anlami — Oncesi vs Sonrasi (kiyas)", expanded=True):
        st.caption(f"Dashboard'da gosterilen her verinin ALGORITMA ONCESI ve SONRASI degeri ile "
                   f"bunun parasal karsiligi. Tasarruf rakamlari makro ({period_lbl}) analizden gelir; "
                   f"yillik projeksiyon ×(365/gun) ile olceklenir.")
        cn = A["costs_naive"]; co = A["costs_opt"]
        idle_pk_n = rated - P["peak_naive"]; idle_pk_o = rated - P["peak_opt"]
        rows_fin = [
            ("Trafo Tepe Yuku (kW)", fmt_num(P['peak_naive']), fmt_num(P['peak_opt']),
             "Tepe ne kadar dusukse demand charge + asim cezasi o kadar az; sozlesme gucu dusurulebilir."),
            ("Bosta Trafo Gucu @tepe (kW)", fmt_num(idle_pk_n), fmt_num(idle_pk_o),
             f"Anma−tepe. Buyuyen bosluk (+{fmt_num(idle_pk_o-idle_pk_n)} kW) = ertelenen trafo yatirimi + ilave EV geliri firsati."),
            ("Trafo Doluluk (%)", f"%{fmt_num(P['peak_naive']/rated*100)}", f"%{fmt_num(P['peak_opt']/rated*100)}",
             "Doluluk dustukce overload/termal risk ve buyume kisiti azalir."),
            ("Enerji Maliyeti (yil, TL)", fmt_tl(cn['energy_cost_tl']*annual_factor_b),
             fmt_tl(co['energy_cost_tl']*annual_factor_b),
             f"Ucuz saatlere kaydirma → dogrudan OPEX tasarrufu: {fmt_tl(A['energy_saving_tl']*annual_factor_b)}/yil."),
            ("Guc Bedeli (yil, TL)", fmt_tl(cn['demand_base_cost_tl']*annual_factor_b),
             fmt_tl(co['demand_base_cost_tl']*annual_factor_b),
             ("EPDK rejimi: sozlesme gucu bazli SABIT (iki tarafta ayni); kazanc, dusuk "
              "sozlesme gucu SECEBILME firsatidir." if P["billing_mode"] == "EPDK"
              else "Tepe-bazli rejim: aylik 15-dk tepe bedeli; tirasama ile dogrudan duser.")),
            ("Guc Asim Cezasi (yil, TL)", fmt_tl(cn['demand_penalty_tl']*annual_factor_b),
             fmt_tl(co['demand_penalty_tl']*annual_factor_b),
             "Sozlesme asiminin cezasi (EPDK ×3, 15-dk ortalama tepe). Algoritma tepeyi esige cekerek sifira yaklastirir."),
            ("Tepe Sicak-Nokta (°C)", fmt_num(th_n['theta_hs_peak']), fmt_num(th_o['theta_hs_peak']),
             "Dusuk sicaklik = ustel olarak yavas yaslanma = ertelenen trafo degisimi."),
            ("30 Yilda Tuketilen Omur (%)", f"%{fmt_num(pj_n['pct_life_horizon'], 2)}", f"%{fmt_num(pj_o['pct_life_horizon'], 2)}",
             f"Korunan omur kesri × trafo bedeli = {fmt_tl(A['thermal_saving_tl'])} ertelenen degisim."),
            ("Ort. SOH Dususu (%)", f"%{fmt_num(soh['final_soh_drop_naive_pct'], 3)}", f"%{fmt_num(soh['final_soh_drop_opt_pct'], 3)}",
             f"Korunan batarya sagligi = {fmt_tl(A['soh_saving_tl'])} geciktirilen batarya degisimi "
             f"({'operator ROI dahil' if A['soh_in_operator_roi'] else 'musteri faydasi, ROI haric'}; gosterge)."),
            ("Ort. Sarj Suresi (dk)", fmt_num(cn['avg_charge_duration_min']), fmt_num(co['avg_charge_duration_min']),
             "Algoritma sureyi uzatabilir (esneklik), ama tamamlanma korunur (S=3 taban ile sinirli)."),
            ("Tamamlanma Orani (%)", f"%{fmt_num(cn['completion_rate']*100, 1)}", f"%{fmt_num(co['completion_rate']*100, 1)}",
             "Araclar %80'e ulasiyor mu? Tasarruf, hizmet kalitesi feda edilmeden saglanir."),
        ]
        fin_df = pd.DataFrame(rows_fin, columns=["Veri (Dashboard)", "Algoritma Oncesi",
                                                 "Algoritma Sonrasi", "Finansal Anlam (Oncesi→Sonrasi)"])
        st.dataframe(fin_df, use_container_width=True, hide_index=True, height=430)
        soh_part = (f" + SOH {fmt_tl(A['operator_soh_saving_tl'])}" if A["soh_in_operator_roi"]
                    else " (SOH musteri faydasi olarak ayri — N5)")
        st.success(
            f"**TOPLAM ({P['cfg_days']} gun): {fmt_tl(A['total_saving_tl'])}** "
            f"= Enerji {fmt_tl(A['energy_saving_tl'])} + Demand/Asim {fmt_tl(A['demand_saving_tl'])} + "
            f"Trafo Omru {fmt_tl(A['thermal_saving_tl'])}{soh_part}. "
            f"**Yillik projeksiyon (mevsim duzeltmeli): {fmt_tl(A['annual_total_saving_tl'])}.**")

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
            mcc[2].metric("Tamamlanma (ort)", f"%{fmt_num(mc['comps'].mean(), 1)}")
            mcc[3].metric("Opt Tepe (ort)", f"{fmt_num(mc['peaks'].mean())} kW")
            cv = mc["totals"].std() / max(mc["totals"].mean(), 1e-9) * 100.0
            figm, axm = plt.subplots(figsize=(10, 3))
            axm.hist(mc["totals"], bins=min(12, len(mc["totals"])), color="#1a73e8", alpha=0.8)
            axm.axvline(mc["totals"].mean(), color="#d93025", ls="--", lw=2, label="ortalama")
            axm.set_xlabel("Toplam Tasarruf (TL)"); axm.set_ylabel("Gun/kosum sayisi")
            axm.legend(fontsize=8); axm.grid(alpha=0.25)
            figm.tight_layout(); st.pyplot(figm); plt.close(figm)
            st.info(f"Degiskenlik katsayisi (CV) = **%{fmt_num(cv, 1)}**. Dusukse (≈<%15) tasarruf "
                    f"sansa az bagli, sonuc saglamdir.")

    # ---- N1: Naive baz cizgisi duyarlilik bandi ----
    with st.expander("🧪 Duyarlilik Bandi — Naive Varsayimi (cesitlilik & olay-tabanli) (N1)", expanded=False):
        st.caption("Naive tepe (ve dolayisiyla ceza/tasarruf) buyuk olcude VARSAYIMA dayanir: "
                   "cesitlilik tavani bir talep tahminidir. Burada ayni kosullar cesitlilik "
                   "0.50/0.60/0.70 ve tavansiz OLAY-TABANLI ust-sinir ile yeniden kosulur; "
                   "sonuclar tek nokta degil BANT olarak okunmalidir.")
        if st.button("🧪 Duyarlilik Bandini Kos"):
            sens_cases = [
                ("Cesitlilik 0.50", dict(diversity=0.50, naive_mode="diversity")),
                ("Cesitlilik 0.60 (varsayilan)", dict(diversity=0.60, naive_mode="diversity")),
                ("Cesitlilik 0.70", dict(diversity=0.70, naive_mode="diversity")),
                ("Olay-tabanli (ust-sinir)", dict(naive_mode="event")),
            ]
            rows_s = []
            prog = st.progress(0.0)
            for j, (nm, over) in enumerate(sens_cases):
                bp = dict(st.session_state["params"]); bp.update(over)
                r = run_simulation(base_mult=1.0, **bp); an = r["analysis"]
                rows_s.append({
                    "Senaryo": nm,
                    "Naive Tepe (kW)": r["peak_naive"],
                    "Naive 15-dk Tepe (kW)": r["peak15_naive"],
                    "Opt Tepe (kW)": r["peak_opt"],
                    "Asim Cezasi Naive (TL)": an["costs_naive"]["demand_penalty_tl"],
                    "Toplam Tasarruf (TL)": an["total_saving_tl"],
                    "Yillik (TL)": an["annual_total_saving_tl"],
                })
                prog.progress((j + 1) / len(sens_cases))
            prog.empty()
            st.session_state["sens"] = pd.DataFrame(rows_s)
        if "sens" in st.session_state:
            sdf = st.session_state["sens"]
            st.dataframe(sdf.style.format({
                "Naive Tepe (kW)": _tr(".0f"), "Naive 15-dk Tepe (kW)": _tr(".0f"),
                "Opt Tepe (kW)": _tr(".0f"), "Asim Cezasi Naive (TL)": _tr(",.0f"),
                "Toplam Tasarruf (TL)": _tr(",.0f"), "Yillik (TL)": _tr(",.0f"),
            }), use_container_width=True, hide_index=True)
            lo_s = float(sdf["Toplam Tasarruf (TL)"].min())
            hi_s = float(sdf["Toplam Tasarruf (TL)"].max())
            st.info(f"📊 **Tasarruf bandi: {fmt_tl(lo_s)} — {fmt_tl(hi_s)}.** Sunumda tek nokta "
                    f"yerine bu bandi (veya P10-P90) raporlamak, naive varsayimina karsi en "
                    f"savunulabilir cercevedir. Olay-tabanli satir UST SINIRDIR (LMS'siz en kotu durum).")

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
                "Kurulu kW": _tr(".0f"), "Opt Tepe (kW)": _tr(".0f"), "Naive Tepe (kW)": _tr(".0f"),
                "Toplam Tasarruf (TL)": _tr(",.0f"), "Tamamlanma (%)": _tr(".1f"),
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
