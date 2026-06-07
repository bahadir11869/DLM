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

from src.config import SimConfig, StationConfig, ScenarioConfig, Weights
from src.optimizer import _waterfill, run_both, SimResult, simulate
from src.financials import thermal_loss_of_life, summarize_costs, build_price_signal
from src.data_generator import build_dataset


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
# termal model
# --------------------------------------------------------------------------- #
def test_thermal_faa_reference_110c():
    # IEEE: 110 C sicak-noktada yaslanma hizlandirma faktoru tam 1 olmali
    theta = 110.0
    faa = np.exp(15000.0 / 383.0 - 15000.0 / (theta + 273.0))
    assert abs(faa - 1.0) < 1e-9


def test_equiv_life_capped_to_design():
    cfg = SimConfig(days=5)
    fac = np.full(cfg.total_minutes, 0.5 * cfg.station.rated_kw)  # hafif yuk
    th = thermal_loss_of_life(cfg, fac)
    assert th["equiv_life_years"] <= cfg.thermal.design_life_years + 1e-9


def test_equiv_life_monotonic_in_load():
    cfg = SimConfig(days=5)
    light = thermal_loss_of_life(cfg, np.full(cfg.total_minutes, 0.6 * cfg.station.rated_kw))
    heavy = thermal_loss_of_life(cfg, np.full(cfg.total_minutes, 1.2 * cfg.station.rated_kw))
    assert heavy["equiv_life_years"] < light["equiv_life_years"]
    assert heavy["pct_life_consumed"] > light["pct_life_consumed"]


# --------------------------------------------------------------------------- #
# demand charge kismi-ay (madde 4)
# --------------------------------------------------------------------------- #
def _const_res(cfg, facility_value):
    T = cfg.total_minutes
    sess = pd.DataFrame({"completed": [True], "charge_duration_min": [30.0]})
    fac = np.full(T, facility_value)
    return SimResult("optimized", np.zeros(T), np.zeros(T), fac, np.zeros(T), np.zeros(T), sess)


def test_demand_charge_fractional_month():
    c30 = SimConfig(days=30)
    c15 = SimConfig(days=15)
    d30 = summarize_costs(c30, _const_res(c30, 1000.0))["demand_base_cost_tl"]
    d15 = summarize_costs(c15, _const_res(c15, 1000.0))["demand_base_cost_tl"]
    # 15 gun = yarim ay -> yari demand bedeli (lineer, yillik projeksiyon tutarli)
    assert abs(d15 - 0.5 * d30) < 1e-6
    # 30 gun = 1 tam ay: tepe(1000) x 90 TL/kW x 1 ay
    assert abs(d30 - 1000.0 * 90.0) < 1e-6


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
