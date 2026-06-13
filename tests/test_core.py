# -*- coding: utf-8 -*-
"""
test_core.py
============
Cekirdek mantik icin hizli birim testleri (pytest).

Kapsam:
  - water-filling toplam korunumu ve tavan kisiti
  - IEEE termal FAA referansi (110 C'de FAA=1) ve omur tavani/monotonluk
  - demand charge kismi-ay (frac) yillik tutarliligi (madde 4)
  - sarj verimi: sebeke gucu > batarya gucu (A4)
  - A1: optimize tepe <= sozlesme gucu

Calistirma:  python -m pytest -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimConfig, StationConfig, ScenarioConfig, Weights, FinancialConfig
from src.optimizer import _waterfill, run_both, SimResult, simulate
from src.financials import (
    thermal_loss_of_life, transformer_life_projection, summarize_costs, build_price_signal,
)
from src.data_generator import build_dataset, ambient_series, seasonal_aging_factor


# --------------------------------------------------------------------------- #
# water-filling
# --------------------------------------------------------------------------- #
def test_waterfill_conservation():
    pmax = np.array([100.0, 50.0, 30.0])
    w = np.array([1.0, 1.0, 1.0])
    a = _waterfill(120.0, pmax, w)
    assert abs(a.sum() - 120.0) < 1e-6
    assert np.all(a <= pmax + 1e-9)


def test_waterfill_cannot_exceed_capacity():
    pmax = np.array([10.0, 10.0])
    a = _waterfill(100.0, pmax, np.array([1.0, 1.0]))
    assert abs(a.sum() - 20.0) < 1e-6          # sum(pmax) ust siniri


def test_waterfill_zero_total():
    pmax = np.array([10.0, 20.0])
    a = _waterfill(0.0, pmax, np.array([1.0, 1.0]))
    assert np.allclose(a, 0.0)


# --------------------------------------------------------------------------- #
# termal model (IEC 60076-7)
# --------------------------------------------------------------------------- #
def test_iec_aging_reference_98c():
    # IEC 60076-7: 98 C sicak-noktada bagil yaslanma hizi V tam 1 olmali
    cfg = SimConfig(days=2)
    th = cfg.thermal
    theta = th.hs_reference_c
    V = th.aging_base ** ((theta - th.hs_reference_c) / th.aging_doubling_k)
    assert abs(V - 1.0) < 1e-12
    # +6 C -> yaslanma ikiye katlanir
    V2 = th.aging_base ** ((theta + th.aging_doubling_k - th.hs_reference_c) / th.aging_doubling_k)
    assert abs(V2 - 2.0) < 1e-9


def test_thermal_monotonic_in_load():
    # Sabit ortam altinda yuk arttikca sicak-nokta ve tuketilen omur artar
    cfg = SimConfig(days=5)
    amb = np.full(cfg.total_minutes, 25.0)
    light = thermal_loss_of_life(cfg, np.full(cfg.total_minutes, 0.6 * cfg.station.rated_kw), amb)
    heavy = thermal_loss_of_life(cfg, np.full(cfg.total_minutes, 1.2 * cfg.station.rated_kw), amb)
    assert heavy["theta_hs_peak"] > light["theta_hs_peak"]
    assert heavy["pct_life_consumed"] > light["pct_life_consumed"]


def test_thermal_increases_with_ambient():
    # Madde 1: ayni yukte daha SICAK ortam -> daha yuksek sicak-nokta ve yaslanma
    cfg = SimConfig(days=5)
    fac = np.full(cfg.total_minutes, 0.9 * cfg.station.rated_kw)
    cool = thermal_loss_of_life(cfg, fac, np.full(cfg.total_minutes, 10.0))
    hot = thermal_loss_of_life(cfg, fac, np.full(cfg.total_minutes, 35.0))
    assert hot["theta_hs_peak"] > cool["theta_hs_peak"]
    assert hot["pct_life_consumed"] > cool["pct_life_consumed"]


def test_life_projection_capped_and_deferred_nonneg():
    # Madde 3: termal-esdeger omur tasarim tavanini asmaz; ertelenen maliyet >= 0
    cfg = SimConfig(days=5)
    amb = np.full(cfg.total_minutes, 25.0)
    th_opt = thermal_loss_of_life(cfg, np.full(cfg.total_minutes, 0.7 * cfg.station.rated_kw), amb)
    th_naive = thermal_loss_of_life(cfg, np.full(cfg.total_minutes, 1.0 * cfg.station.rated_kw), amb)
    lp = transformer_life_projection(cfg, th_opt, th_naive)
    assert lp["proj_opt"]["equiv_life_years"] <= cfg.thermal.design_life_years + 1e-9
    assert lp["deferred_replacement_tl"] >= 0.0
    # daha hafif yuklenen opt, naive'den daha az 30-yil omur tuketir
    assert lp["proj_opt"]["frac_life_horizon"] <= lp["proj_naive"]["frac_life_horizon"] + 1e-12


# --------------------------------------------------------------------------- #
# Gercek Ankara ortam sicakligi (madde 1) + mevsimsel faktor (madde 3)
# --------------------------------------------------------------------------- #
def test_ambient_series_summer_window():
    cfg = SimConfig(days=100)
    amb = ambient_series(cfg)
    assert len(amb) == cfg.total_minutes
    # Mayis-Agustos penceresi: tepe ortam sicakligi yuksek (yaz), >28 C beklenir
    assert amb.max() > 28.0
    assert amb.min() < amb.max()


def test_seasonal_factor_below_one():
    # En kotu (yaz) pencereye gore yil ortalamasi daha serin -> faktor < 1
    cfg = SimConfig(days=100)
    s = seasonal_aging_factor(cfg)
    assert 0.0 < s < 1.0


# --------------------------------------------------------------------------- #
# demand charge kismi-ay (madde 4)
# --------------------------------------------------------------------------- #
def _const_res(cfg, facility_value):
    T = cfg.total_minutes
    sess = pd.DataFrame({"completed": [True], "charge_duration_min": [30.0]})
    fac = np.full(T, facility_value)
    return SimResult("optimized", np.zeros(T), np.zeros(T), fac, np.zeros(T), np.zeros(T), sess)


def test_demand_charge_fractional_month_epdk():
    # N2: varsayilan EPDK rejimi -> guc bedeli SOZLESME GUCU uzerinden sabit
    # (olculen tepeden bagimsiz); kismi ay yine gun oraniyla agirlanir.
    c30 = SimConfig(days=30)
    c15 = SimConfig(days=15)
    d30 = summarize_costs(c30, _const_res(c30, 1000.0))["demand_base_cost_tl"]
    d15 = summarize_costs(c15, _const_res(c15, 1000.0))["demand_base_cost_tl"]
    contracted = c30.financial.contracted_demand_kw
    assert abs(d15 - 0.5 * d30) < 1e-6
    # 30 gun = 1 tam ay: SOZLESME(1300) x 90 TL/kW x 1 ay (tepe 1000 olsa bile)
    assert abs(d30 - contracted * 90.0) < 1e-6


def test_demand_charge_demand_mode_peak_based():
    # Eski ABD-tarzi rejim secilirse olculen (15-dk ortalama) tepe faturalanir.
    cfg = SimConfig(days=30)
    cfg.financial.billing_mode = "DEMAND"
    d = summarize_costs(cfg, _const_res(cfg, 1000.0))
    assert abs(d["demand_base_cost_tl"] - 1000.0 * 90.0) < 1e-6
    # sabit 1000 kW profilde 15-dk ortalama tepe de 1000'dir
    assert abs(d["peak_15min_kw"] - 1000.0) < 1e-6


def test_demand_peak_uses_15min_average():
    # N3: tek dakikalik sivri tepe (3000 kW) 15-dk ortalamada torpulenir; ceza
    # anlik tepeye gore degil 15-dk ortalamaya gore hesaplanmali.
    cfg = SimConfig(days=30)
    T = cfg.total_minutes
    sess = pd.DataFrame({"completed": [True], "charge_duration_min": [30.0]})
    fac = np.full(T, 1000.0)
    fac[100] = 3000.0                      # 1 dakikalik spike (>> sozlesme 1300)
    res = SimResult("optimized", np.zeros(T), np.zeros(T), fac, np.zeros(T), np.zeros(T), sess)
    d = summarize_costs(cfg, res)
    avg15 = (3000.0 + 14 * 1000.0) / 15.0  # ~1133 kW < 1300 -> ceza yok
    assert abs(d["peak_15min_kw"] - avg15) < 1e-6
    assert d["demand_penalty_tl"] == 0.0
    assert d["peak_facility_kw"] == 3000.0  # fiziksel anlik tepe ayrica raporlanir


# --------------------------------------------------------------------------- #
# entegrasyon: verim ve sozlesme tavani
# --------------------------------------------------------------------------- #
def _small_run(activation="always", eff=0.92):
    cfg = SimConfig(
        days=10, seed=7,
        station=StationConfig(n_socket_200=2, n_socket_180=1, n_socket_120=2,
                              charge_efficiency=eff),
        scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=20),
        weights=Weights(0.5, 0.5, 0.5), activation_mode=activation,
    )
    fleet, sessions, base, ptf, smf = build_dataset(cfg, save=False)
    price = build_price_signal(cfg, ptf, smf)
    res = run_both(cfg, sessions, base, price, cfg.weights)
    return cfg, res


def test_efficiency_grid_exceeds_battery():
    cfg, res = _small_run(eff=0.90)
    opt = res["optimized"]
    grid_energy = float(opt.charging_kw.sum())                 # sebeke (faturalanan)
    battery_energy = float(opt.sessions["delivered_kwh"].sum())  # bataryaya giren
    assert grid_energy > battery_energy                         # kayip var
    # oran ~ 1/eff (kucuk tamamlanmama paylari nedeniyle tolerans genis)
    assert grid_energy >= battery_energy / 0.90 - 1.0


def test_opt_peak_under_contracted():
    cfg, res = _small_run()
    co = summarize_costs(cfg, res["optimized"])
    cn = summarize_costs(cfg, res["naive"])
    contracted = cfg.financial.contracted_demand_kw
    assert co["peak_facility_kw"] <= contracted + 1.0      # A1: opt sozlesmeye cekildi
    assert co["demand_penalty_tl"] <= 1.0                  # opt cezasi ~0
    assert cn["peak_facility_kw"] > co["peak_facility_kw"]  # naive daha yuksek tepe


def test_peak_only_does_not_overload():
    # A2: peak_only modunda bile optimize trafoyu/sozlesmeyi asmamali
    cfg, res = _small_run(activation="peak_only")
    co = summarize_costs(cfg, res["optimized"])
    assert co["peak_facility_kw"] <= cfg.financial.contracted_demand_kw + 1.0


# --------------------------------------------------------------------------- #
# madde 9: algoritma ONCESI (naive) bile trafoyu asmamali (overload yok)
# --------------------------------------------------------------------------- #
def test_naive_does_not_overload_default_sizing():
    # Varsayilan istasyon boyutlandirmasi (2/1/1) + cesitlilik faktoru ile naive
    # tepe, trafo etkin anmasini (kVA×cosφ) asmamalidir.
    cfg = SimConfig(
        days=15, seed=11,
        scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=30),
        weights=Weights(0.5, 0.5, 0.5),
    )
    fleet, sessions, base, ptf, smf = build_dataset(cfg, save=False)
    price = build_price_signal(cfg, ptf, smf)
    res = run_both(cfg, sessions, base, price, cfg.weights)
    cn = summarize_costs(cfg, res["naive"])
    assert cn["peak_facility_kw"] <= cfg.station.rated_kw + 1e-6


def test_diversity_caps_naive_coincident_demand():
    # Madde 6: naive esZamanli SEBEKE sarj talebi diversity × kurulu gucu asmamali
    cfg = SimConfig(
        days=15, seed=3,
        scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=40),
        weights=Weights(0.5, 0.5, 0.5),
    )
    fleet, sessions, base, ptf, smf = build_dataset(cfg, save=False)
    price = build_price_signal(cfg, ptf, smf)
    res = run_both(cfg, sessions, base, price, cfg.weights)
    naive = res["naive"]
    cap = cfg.station.diversity_factor * cfg.station.installed_kw
    assert float(naive.charging_kw.max()) <= cap + 1.0


def test_default_sizing_creates_shavable_overload():
    # YENI DEFAULT (2x200=400 kW, baz %75): senaryo ANLAMLI olmali ->
    #   (i) baz yuk TEK BASINA sozlesme gucunu asmaz (EV yokken ceza yok),
    #   (ii) baz + cesitlilikli sarj talebi sozlesme gucunu ASAR (DLM'in
    #        tirasayacagi / cezayi onleyecegi bir asim olusur).
    st = StationConfig(); fin = FinancialConfig()
    base_peak = st.rated_kw * st.base_peak_frac
    assert base_peak < fin.contracted_demand_kw
    assert base_peak + st.diversified_demand_kw > fin.contracted_demand_kw


def test_event_naive_exceeds_diversity_cap():
    # N1: olay-tabanli naive (ust-sinir) cesitlilik tavanini asabilmeli; ayni
    # veriyle diversity-modlu naive tavanin altinda kalmali.
    base_kw = dict(days=10, seed=7,
                   scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=30),
                   weights=Weights(0.5, 0.5, 0.5))
    cfg_div = SimConfig(station=StationConfig(naive_mode="diversity"), **base_kw)
    cfg_evt = SimConfig(station=StationConfig(naive_mode="event"), **base_kw)
    fleet, sessions, base, ptf, smf = build_dataset(cfg_div, save=False)
    price = build_price_signal(cfg_div, ptf, smf)
    nv_div = simulate(cfg_div, sessions, base, price, cfg_div.weights, "naive")
    nv_evt = simulate(cfg_evt, sessions, base, price, cfg_evt.weights, "naive")
    cap = cfg_div.station.diversity_factor * cfg_div.station.installed_kw
    assert float(nv_div.charging_kw.max()) <= cap + 1.0
    assert float(nv_evt.charging_kw.max()) > cap + 1.0   # tavansiz ust-sinir


def test_energy_seasonal_factor_below_one_summer_window():
    # N4: yaz penceresi (Mayis baslangic) icin enerji mevsim faktoru < 1 olmali
    # (yillik projeksiyon, yaz arbitraj makasini yila tasirken torpulenir).
    from src.data_generator import energy_seasonal_factor
    cfg = SimConfig(days=180)
    f = energy_seasonal_factor(cfg)
    assert 0.5 < f < 1.0


def test_dep_uncertainty_degrades_gracefully():
    # N7: cikis saati belirsizligi altinda simulasyon calismali; tamamlanma
    # makul kalmali (S-kati taban guvencesi) ama kehanetli kosumdan iyi olamaz.
    cfg0 = SimConfig(days=10, seed=7,
                     scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=20),
                     weights=Weights(0.5, 0.5, 0.5))
    fleet, sessions, base, ptf, smf = build_dataset(cfg0, save=False)
    price = build_price_signal(cfg0, ptf, smf)
    exact = simulate(cfg0, sessions, base, price, cfg0.weights, "optimized")
    cfg_u = SimConfig(days=10, seed=7,
                      scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=20,
                                              dep_uncertainty_min=60.0),
                      weights=Weights(0.5, 0.5, 0.5))
    noisy = simulate(cfg_u, sessions, base, price, cfg_u.weights, "optimized")
    comp_exact = float(exact.sessions["completed"].mean())
    comp_noisy = float(noisy.sessions["completed"].mean())
    assert comp_noisy > 0.80                      # cokmemeli
    assert comp_noisy <= comp_exact + 0.02        # kehanetten belirgin iyi olamaz


def test_thermal_saving_grows_with_base_load():
    # Baz yuk arttikca sicak-nokta yukselir; yaslanma ustel oldugu icin
    # algoritmanin TERMAL tasarrufu (naive-opt yaslanma maliyeti farki) da artmali.
    from src.financials import thermal_loss_of_life
    from src.data_generator import ambient_series
    cfg = SimConfig(days=10, seed=7,
                    scenario=ScenarioConfig(name="FABRIKA", fleet_size_override=20),
                    weights=Weights(0.5, 0.5, 0.5))
    fleet, sessions, base, ptf, smf = build_dataset(cfg, save=False)
    price = build_price_signal(cfg, ptf, smf)
    amb = ambient_series(cfg)
    savings = []
    for mult in (1.0, 1.2):
        res = run_both(cfg, sessions, base * mult, price, cfg.weights)
        tn = thermal_loss_of_life(cfg, res["naive"].facility_kw, amb)
        to = thermal_loss_of_life(cfg, res["optimized"].facility_kw, amb)
        savings.append(tn["aging_cost_tl"] - to["aging_cost_tl"])
    assert savings[1] > savings[0]


def test_ramp_scales_with_installed_power():
    # Madde 4: ramp kurulu gucun yuzdesinden; alt taban uygulanir
    big = StationConfig(n_socket_200=4, n_socket_180=0, n_socket_120=0)   # 800 kW
    assert abs(big.ramp_kw_per_min - 0.10 * 800.0) < 1e-6
    tiny = StationConfig(n_socket_200=1, n_socket_180=0, n_socket_120=0)  # 200 kW
    assert tiny.ramp_kw_per_min == max(tiny.ramp_floor_kw_per_min, 0.10 * 200.0)
