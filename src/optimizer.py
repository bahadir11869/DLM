# -*- coding: utf-8 -*-
"""
optimizer.py
============
Gercek zamanli, dakika bazli, cok-amacli (Multi-Objective) dinamik yuk dagilimi.

Iki strateji:
  1) "optimized" : alpha/beta/gamma agirlikli, ±60 kW ramp-limitli, C-rate (SOH)
                   korumali, fiyat-bilincli ve TRAFO-KORUMALI akilli dagitim.
                   Toplam yuku (baz+sarj) trafo anma gucunun ustune cikarmaz
                   -> peak-shaving + termal koruma.
  2) "naive"     : "klasik bodoslama" - her arac sokete takildigi an mumkun olan
                   maksimum gucu ceker; SADECE soket donanimiyla sinirlidir.
                   Trafo limitini, ramp'i, C-rate'i ve fiyati YOK sayar
                   -> trafo asiri yuklenir (termal yaslanma), batarya hizli yipranir.

==========================================================================
 ±60 kW RAMPA LIMITININ MATEMATIGI  (Ramp Rate Limit, dP/dt)
==========================================================================
Trafoya binen toplam sarj gucu P_total(t) ani degisemez:

        |P_total(t) - P_total(t-1)|  <=  R        (R = 60 kW / dakika)

Her t dakikasinda HARD CONSTRAINT olarak:

    1) Araclarin "istedigi" hedef toplam guc T_des(t) hesaplanir
       (aciliyet + fiyat sinyaline gore).
    2) Onceki dakikanin gercek gucune (P_prev) gore ramp penceresine kirpilir:
           P_lower = P_prev - R   ;  P_upper = P_prev + R
           T_ramp  = clip(T_des, P_lower, P_upper)
    3) Fiziksel/sebeke tavanlarina kirpilir:
           T(t) = clip(T_ramp, 0, min(C(t), Σ p_max_i))
       C(t) = trafonun o dakika sarja ayirdigi kapasite (yalnizca optimize'da).
    4) T(t), araclara agirlikli water-filling ile dagitilir (p_max_i asilmaz).

==========================================================================
 DINAMIK TRAFO HEADROOM'U  C(t)   (yalnizca optimize)
==========================================================================
    C(t) = max(0, rated_kw * opt_max_loading_pu - base_load(t))
Yani optimize strateji, baz yukun trafoda biraktigi bos kapasiteyi kullanir
ve toplami trafo anma gucunun (opt_max_loading_pu ile) USTUNE cikarmaz.
Bodoslama bu kisiti uygulamaz; toplam yuk trafoyu asabilir (overload).

Performans: dis zaman dongusu (ramp/durum bagimliligi) kacinilmazdir; her
dakikadaki arac-bazli hesaplar NumPy ile vektorizedir, es zamanli arac sayisi
soketle sinirli oldugundan ic islem cok kucuktur.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .config import SimConfig, Weights, HOURS_PER_MINUTE


# --------------------------------------------------------------------------- #
# SoC tabanli sarj egrisi (taper)
# --------------------------------------------------------------------------- #
def _soc_taper(soc: np.ndarray) -> np.ndarray:
    """%70 altinda tam guc; %70->%80 arasi lineer olarak %50'ye iner."""
    return np.where(soc < 0.70, 1.0, np.clip(1.0 - 0.5 * (soc - 0.70) / 0.10, 0.5, 1.0))


# --------------------------------------------------------------------------- #
# Agirlikli su-doldurma (water-filling)
# --------------------------------------------------------------------------- #
def _waterfill(total: float, pmax: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    `total` kW'yi agirliklara orantili, p_max tavanlarini asmadan dagitir.
    Tavana takilanlarin artigi kalanlara yeniden dagitilir. Tam vektorize.
    """
    n = pmax.shape[0]
    alloc = np.zeros(n, dtype=np.float64)
    if n == 0 or total <= 0.0:
        return alloc
    total = min(total, float(pmax.sum()))
    free = np.ones(n, dtype=bool)
    w = np.where(weight > 0, weight, 0.0).astype(np.float64)

    for _ in range(n + 2):
        remaining = total - alloc.sum()
        if remaining <= 1e-9:
            break
        wfree = w[free].sum()
        if wfree <= 0:
            idx = np.where(free)[0]
            room = pmax[idx] - alloc[idx]
            if room.sum() > 0:
                alloc[idx] += np.minimum(room / room.sum() * remaining, room)
            break
        share = np.zeros(n)
        share[free] = w[free] / wfree * remaining
        new_alloc = alloc + share
        over = free & (new_alloc >= pmax)
        if not over.any():
            alloc = new_alloc
            break
        alloc[over] = pmax[over]
        free[over] = False
    return np.minimum(alloc, pmax)


# --------------------------------------------------------------------------- #
# Sonuc kabi
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    strategy: str
    charging_kw: np.ndarray      # dakika bazli toplam sarj gucu (kW)
    base_load_kw: np.ndarray     # dakika bazli baz yuk (kW)
    facility_kw: np.ndarray      # baz + sarj (kW)  -> termal model girdisi
    headroom_kw: np.ndarray      # optimize icin C(t) (kW)
    price_tl_kwh: np.ndarray     # dakika bazli enerji fiyati (TL/kWh)
    sessions: pd.DataFrame       # oturum bazli sonuclar (vehicle_id, day, ...)


# --------------------------------------------------------------------------- #
# Ana simulasyon
# --------------------------------------------------------------------------- #
def simulate(
    cfg: SimConfig,
    sessions: pd.DataFrame,
    base_load: np.ndarray,
    price_tl_kwh: np.ndarray,
    weights: Weights,
    strategy: str = "optimized",
) -> SimResult:
    st = cfg.station
    T = len(base_load)
    dt = HOURS_PER_MINUTE
    target = st.target_soc
    ramp = st.ramp_kw_per_min
    stretch = max(st.max_stretch_factor, 1.0)   # madde 3: maks sarj uzatma katsayisi
    rated = st.rated_kw
    max_load = st.opt_max_loading_pu
    eff = float(min(max(st.charge_efficiency, 0.5), 1.0))  # A4: sebeke->batarya verimi
    is_opt = (strategy == "optimized")

    # A1: OPTIMIZE HEDEF TAVANI = min(fiziksel trafo, sozlesme gucu).
    # Ekonomik ceza esigi (sozlesme gucu, 1400) trafo anmasindan (1600) dusukse,
    # optimize tepeyi SOZLESME gucune cekmeye calisir -> guc asim cezasi sifirlanir.
    # (Naive bu tavani yok sayar; fiziksel olarak trafoyu da asabilir -> overload.)
    cap_kw = rated * max_load
    contracted = float(cfg.financial.contracted_demand_kw)
    if contracted > 0:
        cap_kw = min(cap_kw, contracted)
    is_factory = cfg.scenario.is_factory
    activation_mode = cfg.activation_mode
    peak_threshold = cfg.peak_loading_threshold

    s = sessions.reset_index(drop=True)
    n = len(s)
    cap = s["capacity"].to_numpy(np.float64)
    hw = s["dc_max"].to_numpy(np.float64)
    arr = s["arr_global"].to_numpy(np.int64)
    dep = s["dep_global"].to_numpy(np.int64)
    eneed = s["energy_need_kwh"].to_numpy(np.float64)   # oturum toplam enerji ihtiyaci (kWh)
    soc = s["arrival_soc"].to_numpy(np.float64).copy()

    delivered = np.zeros(n)
    stress_thr = np.zeros(n)
    start_min = np.full(n, -1, np.int64)
    finish_min = np.full(n, -1, np.int64)

    sock_cap = np.sort(np.array(st.socket_list(), dtype=np.float64))[::-1]
    n_sock = sock_cap.shape[0]
    sock_veh = np.full(n_sock, -1, np.int64)

    # C-rate (SOH) tavani: beta=0 -> crate_high, beta=1 -> crate_low
    crate_cap = st.crate_high - weights.beta * (st.crate_high - st.crate_low)

    pmin_price, pmax_price = float(price_tl_kwh.min()), float(price_tl_kwh.max())
    prange = max(pmax_price - pmin_price, 1e-9)

    # Fiyat kumulatif toplami: bir aracin kalan penceresindeki ORTALAMA fiyati
    # O(1)'de hesaplamak icin (gelecek-farkindali maliyet sinyali).
    cumprice = np.concatenate([[0.0], np.cumsum(price_tl_kwh)])

    charging_out = np.zeros(T)
    headroom_out = np.zeros(T)

    order = np.argsort(arr)
    ptr = 0
    queue: deque = deque()
    prev_total = 0.0

    for t in range(T):
        # (a) gelisleri kuyruga al
        while ptr < n and arr[order[ptr]] <= t:
            i = order[ptr]; ptr += 1
            if t < dep[i] and soc[i] < target - 1e-6:
                queue.append(i)

        # (b) biten/kalkan araclari soketten birak
        for ssi in range(n_sock):
            i = sock_veh[ssi]
            if i < 0:
                continue
            if soc[i] >= target - 1e-6:
                finish_min[i] = t; sock_veh[ssi] = -1
            elif t >= dep[i]:
                finish_min[i] = t; sock_veh[ssi] = -1   # tamamlanamadi

        # (c) kuyrukta deadline gecmisleri at
        if queue:
            queue = deque(i for i in queue if t < dep[i])

        # (d) bos soketlere UYGUN SOKET atamasi
        free_sockets = [ssi for ssi in range(n_sock) if sock_veh[ssi] < 0]
        while queue and free_sockets:
            i = queue.popleft()
            need = hw[i]
            best, best_key = None, None
            for ssi in free_sockets:
                c = sock_cap[ssi]
                key = (0, c - need) if c >= need else (1, need - c)
                if best_key is None or key < best_key:
                    best_key, best = key, ssi
            sock_veh[best] = i
            free_sockets.remove(best)
            if start_min[i] < 0:
                start_min[i] = t

        # (e) aktif araclarin vektorel hesabi
        occ = np.where(sock_veh >= 0)[0]
        if occ.size == 0:
            prev_total = 0.0
            charging_out[t] = 0.0
            headroom_out[t] = max(0.0, rated * max_load - base_load[t])
            continue

        veh = sock_veh[occ]
        soc_a = soc[veh]
        cap_a = cap[veh]
        hw_a = np.minimum(hw[veh], sock_cap[occ])          # arac VE soket limiti
        energy_rem = np.maximum(0.0, (target - soc_a) * cap_a)

        # S-KATI EFEKTIF DEADLINE (madde 3 - sure/yuzde tavani):
        # Aracin sarj suresini, tahmini TAM-GUC (naive) suresinin EN FAZLA S katiyla
        # sinirla. Boylece "gecikme yuzdesi" ~ (S-1)x100 ile sinirli kalir; algoritma
        # trafo bos olsa bile gucu sonsuza kadar yayip sureyi 4-5 katina cikaramaz.
        #   t_naive ≈ enerji / (min(dc,soket)*0.85)   (taper dahil yaklasik tam-guc)
        #   eff_dep = baslangic + S * t_naive   (gercek cikistan once)
        avg_full_pow = np.minimum(hw[veh], sock_cap[occ]) * 0.85
        t_naive_min = eneed[veh] / np.maximum(avg_full_pow, 1e-9) / dt   # dakika
        sm = np.where(start_min[veh] >= 0, start_min[veh], t)
        eff_dep = np.minimum(dep[veh], sm + np.ceil(stretch * t_naive_min)).astype(np.int64)
        eff_dep = np.maximum(eff_dep, t + 1)

        # ACILIYET GUCU (eff_dep'e yetisme): aracin %80'e ulasmasi icin gereken
        # asgari guc. ALPHA (sarj suresi onceligi) bu pencereyi sikistirir:
        #   safety = 0.85 (alpha=0, rahat) ... 0.50 (alpha=1, erken bitir)
        # Kucuk pencere -> daha yuksek aciliyet gucu -> daha hizli sarj.
        safety = 0.85 - 0.35 * weights.alpha
        time_left = np.maximum(eff_dep - t, 1)
        time_left_eff = np.maximum(safety * time_left, 1.0)
        p_need = energy_rem / (time_left_eff * dt)

        p_energy = energy_rem / dt
        p_taper = hw_a * _soc_taper(soc_a)
        p_hard = np.maximum(np.minimum(p_taper, p_energy), 0.0)  # C-rate'i yok sayar

        # ALGORITMA DEVREYE GIRME POLITIKASI (madde 6 + A2):
        #   - is_opt icin TRAFO/SOZLESME KORUMASI ve ramp DAIMA aciktir; bu fiziksel
        #     guvenliktir, kapatilamaz (aksi halde "peak_only"da gece filo dalgasinda
        #     baz dusukken sarj trafoyu asardi - A2 hatasi).
        #   - FIYAT/SOH OPTIMIZASYONU (price_smart) ise aktivasyona baglidir:
        #       "always"    -> her dakika.
        #       "peak_only" -> sadece (baz + TALEP EDILEN sarj) doluluk esigi astiginda;
        #                      altinda trafo-korumali ama fiyat/SOH'suz "acgozlu doldur".
        demanded_grid = float(p_hard.sum()) / eff     # talep edilen sarjin sebeke karsiligi
        loading = (base_load[t] + demanded_grid) / rated
        price_smart = is_opt and (
            activation_mode == "always" or loading >= peak_threshold
        )
        headroom_out[t] = max(0.0, rated * max_load - base_load[t])

        # GUC TABANI (madde 3 - geri besleme): anlik guc, aracin tam-guc kabiliyetinin
        # 1/S'inin altina inmesin -> sarj suresi <= S x minimum (DC avantaji korunur).
        p_floor = p_hard / stretch

        if is_opt:
            if price_smart:
                # SOH-dostu yumusak C-rate tavani + guc tabani; ACILIYET gecersiz kilabilir.
                p_soft = np.minimum(p_hard, crate_cap * cap_a)
                need_cap = np.minimum(p_hard, p_need)
                pmax = np.clip(np.maximum.reduce([p_soft, need_cap, p_floor]), 0.0, p_hard)
            else:
                # Optimizasyon-pasif ama TRAFO-KORUMALI: C-rate/fiyat yok, sadece tavan altinda doldur.
                pmax = p_hard
            # Trafo/sozlesme headroom'u SEBEKE tarafindadir; batarya tarafina verimle cevrilir
            # (alloc batarya gucudur; sebeke = alloc/eff). Boylece baz+sarj(sebeke) <= cap_kw.
            C_t = max(0.0, cap_kw - base_load[t]) * eff
        else:
            # Bodoslama (naive): trafo limiti, sozlesme, ramp ve C-rate YOK -> overload mumkun.
            pmax = p_hard
            C_t = float(pmax.sum())

        if is_opt and price_smart:
            # ============================================================== #
            #  HEDEF TOPLAM GUC  T_des = (ACIL) + (FIRSATCI)
            # ============================================================== #
            # 1) ACIL guc: her aracin deadline'a yetismesi icin SART olan kisim.
            #    Bu daima verilir (servis garantisi). ALPHA, p_need uzerinden
            #    bu kismi buyutur -> daha hizli sarj.
            # ACIL + GARANTILI TABAN: deadline gucu (p_need) ile guc tabani
            # (p_floor) her arac icin GARANTI edilir (asagida once tahsis edilir).
            p_urgent = np.minimum(np.maximum(p_need, p_floor), pmax)

            # 2) FIRSATCI guc: acil olmayan, "istege bagli" sarj. ONEMLI: bu islem
            #    araci BASKA SAATE TASIMAZ; aracin geliş/cikis zamani sabittir. Sadece
            #    aracin KENDI fis-takili penceresi icinde gucu pahali dakikalardan
            #    ucuz dakikalara dogru SEKILLENDIRIR (load shaping). Her arac icin
            #    GELECEK-FARKINDALI fiyat sinyali:
            #       future_mean_i = aracin KALAN penceresindeki ortalama fiyat
            #       rel_i = (future_mean_i - fiyat_t) / fiyat_araligi
            #               > 0  -> "su an, kendi penceresinin ortalamasindan UCUZ" (yuklen)
            #               < 0  -> "su an pahali, opsiyonel gucu KIS" (pencere ici bekle)
            #       opp_frac_i = ALPHA*0.45        (proaktif taban -> tamamlanma)
            #                  + GAMMA*1.5*rel_i    (pencere ici ucuz dakikaya bindirir)
            #    GAMMA arttikca opsiyonel guc, pencere icindeki pahali dakikalardan
            #    UCUZ dakikalara kaydirilir -> enerji maliyeti DUSER; tamamlanma ~sabit.
            dep_v = np.minimum(eff_dep, T)        # fiyat penceresi de S-kati ile sinirli
            win = np.maximum(dep_v - t, 1)
            future_mean = (cumprice[dep_v] - cumprice[t]) / win
            rel = (future_mean - price_tl_kwh[t]) / prange
            opp_frac = np.clip(weights.alpha * 0.45 + weights.gamma * 1.5 * rel + 0.05, 0.0, 1.0)
            p_opp = (pmax - p_urgent) * opp_frac

            T_des = float((p_urgent + p_opp).sum())

            # 3) ±60 kW RAMP LIMITI
            T_ramp = min(max(T_des, prev_total - ramp), prev_total + ramp)
            # 4) Fiziksel/trafo tavanlari
            T_t = float(np.clip(T_ramp, 0.0, min(C_t, float(pmax.sum()))))

            # 5) ONCELIKLI (ESIT OLMAYAN) water-filling dagitimi (madde 4):
            #    Toplam guc T_t araclara ESIT degil, ONCELIK SKORUNA gore dagitilir.
            #    Skor uc bilesenden olusur:
            #      (a) ACILIYET (urg_norm): deadline'a en yakin / en cok guc gereken
            #          araclar oncelikli (ALPHA ile agirliklanir).
            #      (b) DOLULUK IHTIYACI (soc_need): SoC'si en DUSUK (en bos) araclar
            #          oncelikli -> bos araclar once toparlanir.
            #      (c) SOH (cap_norm): BETA ile, buyuk bataryali (ayni guçte dusuk
            #          C-rate) araclara goreli oncelik -> filo C-rate'i duser.
            #    Tavana (pmax) ulasan aracin artigi digerlerine yeniden dagitilir.
            urg_norm = p_need / (p_need.max() + 1e-9)
            soc_need = (target - soc_a) / target          # 1=bos, 0=dolu
            cap_norm = cap_a / (cap_a.max() + 1e-9)
            w = (0.6 + weights.alpha) * urg_norm + 0.8 * soc_need \
                + weights.beta * cap_norm + 1e-3
            # ONCE garantili taban (acil + guc tabani) tahsis edilir; KALAN
            # firsatci guc oncelik agirligiyla dagitilir. Boylece guc tabani
            # fiilen GARANTI olur (trafo/ramp tavani yettigi surece) ve sarj
            # suresi S katiyla sinirli kalir. Tavan dar oldugunda (ramp/trafo
            # kisiti) tabana inilemezse oncelik agirligina gore tahsis edilir.
            g_sum = float(p_urgent.sum())
            if T_t <= g_sum + 1e-9:
                # Tavan dar (contention): garantili tabani TABANA ORANTILI dagit ki
                # hicbir arac disproporsiyonel ac kalmasin -> uzatma yuzdeleri TEKDUZE
                # ve sinirli kalir (oncelik agirligiyla dagitim birkac araci asiri
                # yavaslatip %600'lere cikariyordu).
                alloc = _waterfill(T_t, pmax, p_urgent)
            else:
                extra = _waterfill(T_t - g_sum, pmax - p_urgent, w)
                alloc = p_urgent + extra
        elif is_opt:
            # TRAFO-KORUMALI ama OPTIMIZASYON-PASIF (peak_only, esik altinda - A2):
            # fiyat/SOH yok ama trafo/sozlesme tavani ve ramp UYGULANIR (overload yok).
            T_des = float(pmax.sum())
            T_ramp = min(max(T_des, prev_total - ramp), prev_total + ramp)
            T_t = float(np.clip(T_ramp, 0.0, min(C_t, float(pmax.sum()))))
            alloc = _waterfill(T_t, pmax, pmax)
        else:
            # BODOSLAMA / naive: herkes maksimum (istasyon-limitli),
            # ramp/fiyat/trafo/C-rate YOK. Paylasim yine tavanlara orantilidir.
            T_t = float(pmax.sum())
            alloc = _waterfill(T_t, pmax, pmax)

        # (f) entegrasyon. alloc = BATARYA tarafi guc (kW); SEBEKE = alloc/eff (A4).
        energy = alloc * dt                      # bataryaya giren enerji (kWh)
        soc[veh] = soc_a + energy / cap_a
        delivered[veh] += energy
        crate = alloc / cap_a                    # batarya C-rate (SOH stresi)
        stress_thr[veh] += energy * (1.0 + cfg.financial.soh_crate_k * crate ** 2)

        prev_total = float(alloc.sum())          # batarya tarafi (ramp referansi)
        charging_out[t] = prev_total / eff       # SEBEKE tarafi (trafo yuku + faturalanan)

    # --- Oturum sonuc tablosu ---
    res = s.copy()
    res["final_soc"] = soc
    res["delivered_kwh"] = delivered
    res["stress_throughput_kwh"] = stress_thr
    res["completed"] = soc >= target - 1e-3
    res["start_min"] = start_min
    res["finish_min"] = finish_min
    dur = np.where((finish_min >= 0) & (start_min >= 0), finish_min - start_min, np.nan).astype(float)
    res["charge_duration_min"] = dur

    facility = base_load + charging_out
    return SimResult(
        strategy=strategy,
        charging_kw=charging_out,
        base_load_kw=base_load,
        facility_kw=facility,
        headroom_kw=headroom_out,
        price_tl_kwh=price_tl_kwh,
        sessions=res,
    )


def run_both(cfg, sessions, base_load, price_tl_kwh, weights) -> Dict[str, SimResult]:
    """Optimize ve bodoslama stratejilerini ayni veriyle calistirir."""
    return {
        "optimized": simulate(cfg, sessions, base_load, price_tl_kwh, weights, "optimized"),
        "naive": simulate(cfg, sessions, base_load, price_tl_kwh, weights, "naive"),
    }
