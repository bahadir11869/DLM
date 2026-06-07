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
START_DATE = _dt.date(2025, 1, 1)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- #
# Onbellekli simulasyon
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_simulation(scenario, pricing_mode, days, seed, regen_token, fleet_size,
                   n200, n180, n120, alpha, beta, gamma, activation_mode,
                   use_epias, epias_user, epias_pass, base_mult):
    eff_seed = int(seed) + int(regen_token) * 7919
    cfg = SimConfig(
        days=days, seed=eff_seed,
        station=StationConfig(n_socket_200=n200, n_socket_180=n180, n_socket_120=n120),
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
    return {
        "cfg_days": days, "rated_kw": cfg.station.rated_kw,
        "pricing_mode": pricing_mode, "ptf_source": ptf_source,
        "fleet_size": len(fleet), "n_sessions": len(sessions),
        "installed_kw": float(sum(cfg.station.socket_list())),
        "base_min_kw": float(base_load.min()), "base_max_kw": float(base_load.max()),
        "max_install_kw": float(cfg.station.rated_kw - base_load.min()),
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
st.sidebar.caption("Her istasyon TEK soketlidir (ayni anda tek arac).")
c1, c2, c3 = st.sidebar.columns(3)
n200 = c1.number_input("200 kW", 0, 12, 2)
n180 = c2.number_input("180 kW", 0, 12, 1)
n120 = c3.number_input("120 kW", 0, 12, 2)

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
st.caption("1600 kVA Trafo · Tek-soketli Karisik DC Istasyonlar · PTF/SMF & IEEE C57.91 Termal Model")

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
3. **Trafo + ±60 kW ramp:** Toplam (baz+sarj) yuk trafo anmasini asamaz; dakikalik
   degisim ±60 kW.
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
- **Trafo Termal (IEEE C57.91):** sicak-nokta θH'den FAA=exp(15000/383−15000/(θH+273));
  Tuketilen omur=Σ FAA·Δt; **Yaslanma Maliyeti** = (omur kesri)×Trafo (4.000.000 TL).
- **SOH:** stres_kWh=Σ E·(1+0.6·C-rate²); SOH kaybi=stres/(kapasite·1500)·%20;
  **Geciktirilen Degisim** = Σ (kayip farki)×kapasite×4500 TL/kWh.
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

# Kurulum guardrail (madde 3)
installed = P["installed_kw"]; maxinstall = P["max_install_kw"]
if installed > maxinstall:
    st.warning(f"⚠️ Kurulu istasyon gucu **{installed:.0f} kW** > gece headroom "
        f"**{maxinstall:.0f} kW** ({P['rated_kw']:.0f} − min baz {P['base_min_kw']:.0f}). "
        f"Bu kurulum ancak **dinamik yuk yonetimi** ile guvenlidir; bodoslama gunduz "
        f"trafoyu asar (overload).")
else:
    st.success(f"✅ Kurulu istasyon gucu **{installed:.0f} kW** ≤ gece headroom "
        f"**{maxinstall:.0f} kW**. Statik guvenli kurulum.")

top = st.columns(4)
top[0].metric("Toplam Tasarruf (100 gun)", fmt_tl(A["total_saving_tl"]))
top[1].metric("Yillik Projeksiyon", fmt_tl(A["annual_total_saving_tl"]))
top[2].metric("Trafo Tepe (Once→Sonra)",
              f"{P['peak_naive']:.0f}→{P['peak_opt']:.0f} kW")
top[3].metric("Filo / Oturum", f"{P['fleet_size']} / {P['n_sessions']}")

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
        g = st.columns(5)
        show_base = g[0].checkbox("Baz Yuk", True)
        show_naive = g[1].checkbox("Algoritma Oncesi", True)
        show_opt = g[2].checkbox("Algoritma Sonrasi", True)
        show_rated = g[3].checkbox("Trafo Anma", True)
        show_price = g[4].checkbox("Fiyat Egrisi", True)
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

    with st.expander("⚡ Sarj Gucu Egrisi (±60 kW ramp etkisi)", expanded=False):
        fig2, axc = plt.subplots(figsize=(12, 3.6))
        axc.plot(x, chg_naive, color="#d93025", lw=2, label="Bodoslama (sicrayan)")
        axc.plot(x, chg_opt, color="#1e8e3e", lw=2, label="Optimize (rampa ile yumusak)")
        axc.set_xlabel("Saat"); axc.set_ylabel("Sarj Gucu (kW)")
        axc.set_xlim(0, 24); axc.set_xticks(range(0, 25, 2)); axc.grid(alpha=0.25)
        axc.legend(fontsize=8); fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

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

    with st.expander("🔥 Trafo Termal Omur Tuketimi (IEEE C57.91 benzeri)", expanded=True):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tepe Sicak-Nokta (Once→Sonra)",
                  f"{th_n['theta_hs_peak']:.0f}→{th_o['theta_hs_peak']:.0f} °C",
                  help="110°C uzeri yalitimda hizlandirilmis yaslanma baslar.")
        k2.metric("Esdeger Trafo Omru (Once→Sonra)",
                  f"{th_n['equiv_life_years']:.0f}→{th_o['equiv_life_years']:.0f} yil",
                  help="Termal yaslanmaya gore esdeger omur. Dusuk yuklenmede termal "
                       "yaslanma ihmal edilebilir oldugundan deger, fiziksel tasarim omru "
                       "(~30 yil) tavaniyla kirpilir; gercek omru nem/busing/OLTC gibi "
                       "etkenler sinirlar. Iki senaryo arasindaki fark anlamlidir.")
        k3.metric("Tuketilen Omur (Once→Sonra)",
                  f"%{th_n['pct_life_consumed']:.4f}→%{th_o['pct_life_consumed']:.4f}")
        k4.metric("Onlenen Yaslanma (100g / yillik)", fmt_tl(A["thermal_saving_tl"]),
                  f"yillik ~{fmt_tl(A['thermal_saving_annual_tl'])}")
        if th_n["theta_hs_peak"] >= 105.0:
            st.warning(f"⚠️ Algoritmasiz tepe sicak-nokta **{th_n['theta_hs_peak']:.0f}°C** "
                       f"ile 110°C sinirini zorluyor; optimize **{th_o['theta_hs_peak']:.0f}°C**'ye cekiyor.")
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
        st.caption("Her arac icin SOH dususu, toplam sarj suresi ve EN COK uzatilan "
                   "tekil oturum (dakika + %).")
        st.dataframe(
            soh["table"].style.format({
                "SOH Dususu Algoritmali (%)": "{:.4f}", "SOH Dususu Algoritmasiz (%)": "{:.4f}",
                "SOH Korunan (puan)": "{:.4f}",
                "Toplam Sarj Suresi Algoritmali (dk)": "{:,.0f}",
                "Toplam Sarj Suresi Algoritmasiz (dk)": "{:,.0f}",
                "Maks Sarj Uzatma (dk)": "{:.0f}", "Maks Sarj Gecikmesi (%)": "{:.1f}",
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
    f[2].metric("Trafo Omru Tasarrufu", fmt_tl(A["thermal_saving_tl"]))
    f[3].metric("SOH (Batarya) Tasarrufu", fmt_tl(A["soh_saving_tl"]))
