# -*- coding: utf-8 -*-
"""
config.py
=========
Tum simulasyon icin paylasilan sabitler ve yapilandirma nesneleri.

Kapsanan konular:
  - Donanim: 1600 kVA ana trafo (cosφ=0.95 -> 1520 kW etkin), karisik DC soket
    parki (200/180/120 kW). Cesitlilik (esZamanlilik) faktoru - madde 6.
  - Baz yuk: trafo etkin anmasinin %60'ina (≈912 kW) ulasan tesis profili (madde 9).
  - Fiyatlandirma: (a) PTF/SMF (EPIAS piyasa) - buyuk fabrika olcegi,
                   (b) 3-zamanli sanayi tarifesi - daha kucuk tesis olcegi.
  - Optimizasyon kisitlari: ramp (kurulu gucun %10/dk), %80 SoC bitis, C-rate tavani.
  - Trafo Termal Omur modeli: IEC 60076-7:2018 (madde 2) + gercek Ankara
    sicakliklari (madde 1) + 30 yillik omur ekstrapolasyonu (madde 3).
  - Batarya SOH ve finansal (ROI) parametreleri.

Tum degerler gercek dunya verilerine ve ilgili standartlara yakin secilmistir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Zaman ekseni
# --------------------------------------------------------------------------- #
MINUTES_PER_DAY: int = 1440           # Simulasyon dakika bazli calisir.
HOURS_PER_MINUTE: float = 1.0 / 60.0  # dt = 1/60 saat (kW -> kWh donusumu)
DAYS_PER_MONTH: int = 30              # Slider AY -> GUN donusumu (madde 1)
MONTHS_DEFAULT: int = 6               # Varsayilan simulasyon suresi (ay)
DAYS_DEFAULT: int = MONTHS_DEFAULT * DAYS_PER_MONTH   # 180 gun (~6 ay)


# --------------------------------------------------------------------------- #
# Donanim (1600 kVA trafo + karisik DC soket parki)
# --------------------------------------------------------------------------- #
@dataclass
class StationConfig:
    """Sarj istasyonu donanim yapilandirmasi."""

    # Ana trafo: 1600 kVA gorunur guc.
    transformer_kva: float = 1600.0

    # ORTALAMA GUC FAKTORU (madde 6): trafo etkin (aktif) gucu = kVA x cosφ.
    # Modern DC hizli sarj cihazlari aktif PFC ile cosφ≈0.98-0.99 calisir; tesis
    # baz yuku (motor/aydinlatma/HVAC) ile birlikte TESIS ORTALAMA guc faktoru
    # tipik olarak ~0.95'tir (EPDK reaktif-ceza esigi cosφ≥0.90'in ustunde tutulur).
    # rated_kw = 1600 x 0.95 = 1520 kW etkin guc.
    power_factor: float = 0.95

    # BAZ YUK TEPE ORANI (madde 9): tesis baz yuk tepesi trafonun bu oranina ulasir.
    # KUCUK SARJ PARKI (default 2x200=400 kW) ile senaryoyu ANLAMLI tutmak icin baz
    # yuk yukseltilmistir: %75 -> 0.75 x 1520 = 1140 kW tepe.
    #   - baz TEK BASINA (1140) < sozlesme gucu (1300) -> EV olmadan ceza yok.
    #   - baz + cesitlilikli sarj talebi (1140 + 240 ≈ 1380) > sozlesme (1300)
    #     -> ALGORITMASIZ ceza olusur (DLM'in tirasayacagi anlamli asim).
    #   - 1380 < trafo anmasi (1520) -> fiziksel overload YOK; optimize 1300'e kirpar.
    base_peak_frac: float = 0.75          # ≈ 1140 kW tepe (kucuk parkta senaryoyu canli tutar)

    # Optimize stratejinin trafoyu yuklemesine izin verdigi ust sinir (p.u.).
    opt_max_loading_pu: float = 1.00      # optimize: trafoyu asma

    # KARISIK DC ALTYAPI (madde 9): istasyon sayilari, CESITLILIK (esZamanlilik)
    # FAKTORU devredeyken algoritma-oncesi toplam istasyon talebinin trafo
    # anmasinin %20-%30 araliginda kalmasini saglayacak sekilde secilir.
    #   kurulu guc = 2x200 + 1x180 + 1x120 = 700 kW
    #   cesitlilikli talep = 700 x 0.60 = 420 kW ≈ trafo anmasinin (1520) %27.6'si
    # Baz tepe (%60) + cesitlilikli istasyon (%27.6) = %87.6 < %100 -> overload YOK.
    n_socket_200: int = 2     # 2 x 200 kW   (DEFAULT kurulum: toplam 400 kW)
    n_socket_180: int = 0     # 1 x 180 kW
    n_socket_120: int = 0     # 1 x 120 kW

    # CESITLILIK / ESZAMANLILIK (DIVERSITY) FAKTORU (madde 6):
    # IEC 60364-7-722: bir Yuk Yonetim Sistemi (LMS) YOKSA tasarim cesitlilik
    # faktoru 1 alinir (tum noktalar ayni anda tam guc cekebilir). Pratik TALEP
    # TAHMININDE ise, bir DC sarj kumesinin gercek esZamanli talebi kuruludan
    # dusuktur (varislerin/taper'in dagilimi). Bu modelde:
    #   - ALGORITMA-ONCESI (naive, LMS yok): esZamanli sarj talebi
    #     diversity_factor x kurulu_guc ile sinirlanir (gercekci talep tahmini).
    #   - ALGORITMA (optimize) IEC 60364-7-722'deki LMS'in karsiligidir; talebi
    #     aktif olarak trafo headroom'u icinde yonetir.
    diversity_factor: float = 0.60

    # ALGORITMA-ONCESI (naive) TALEP MODELI (N1 - baz cizgisi durustlugu):
    #   "diversity" -> esZamanli talep diversity_factor x kurulu_guc ile sinirli
    #                  (talep TAHMINI; naive tepe buyuk olcude bu varsayima baglidir).
    #   "event"     -> tavansiz OLAY-TABANLI ust-sinir: tum soketler taper'a gore
    #                  serbest ceker (LMS yokken fiziksel olarak mumkun olan en kotu
    #                  durum; overload MUMKUNDUR). Iki mod birlikte, naive tepenin
    #                  belirsizlik BANDINI verir (sonuclar tek noktaya degil banda
    #                  gore raporlanmalidir).
    naive_mode: str = "diversity"

    # RAMPA HIZI KISITI (madde 4): EV sarj cihazlari icin "kW/dk" cinsinden ZORUNLU
    # bir standart YOKTUR; cihazin kendisi ISO 15118 / IEC 61851 ile saniyeler
    # icinde rampa yapabilir. Sinir, SAHA EMS'inin GUC KALITESI (gerilim
    # dalgalanmasi / flicker) icin koydugu bir yumusatma tercihidir
    # (IEC 61000-3-3 / IEC 61000-3-11). Bu nedenle ramp, KONTROL EDILEBILIR sarj
    # gucunun bir YUZDESI olarak ifade edilir (mutlak kW yerine olceklenebilir):
    #   ramp_kw/dk = ramp_frac_per_min x kurulu_istasyon_gucu  (alt taban ile)
    # %10/dk -> 700 kW kuruluda 70 kW/dk (eski sabit ~60 kW/dk ile uyumlu mertebede).
    ramp_frac_per_min: float = 0.10
    ramp_floor_kw_per_min: float = 30.0   # cok kucuk kurulumlarda alt taban

    # SARJ VERIMI (madde 7): sebekeden cekilen gucun bataryaya giren orani.
    # DC hizli sarj zinciri (AC/DC donusum + isil + yardimci) tipik verimi
    # ~%92-95'tir; ORTALAMA %93 alinir. Sebeke (trafo+fatura) gucu = batarya
    # gucu / verim; yani fatura ve trafo yuku bataryaya gireninden buyuktur.
    charge_efficiency: float = 0.93

    # SABIT SARJ BITIS KURALI: tum araclar %80 SoC'de biter.
    target_soc: float = 0.80

    # MAKS SARJ UZATMA KATSAYISI (S) - madde 3:
    # Algoritma bir araci, tam-guc (en kotu senaryo / algoritmasiz) ile gereken
    # sureye gore EN FAZLA S katina kadar uzatabilir. Anlik verilen guc, aracin
    # tam-guc kabiliyetinin 1/S'inin altina dusurulmez (guc tabani = p_hard / S).
    # Boylece DC hizinin avantaji korunur; aksi halde sure 400+ dk'ya cikip
    # araba AC ile sarj olmustan farksiz kalir.
    max_stretch_factor: float = 3.0

    # SOH/C-rate guvenli sarj penceresi (optimize moda ozel):
    #   izin verilen max C-rate = crate_high - beta*(crate_high - crate_low)
    crate_high: float = 2.5   # beta=0 -> agresif
    crate_low: float = 0.8    # beta=1 -> bataryayi koru

    @property
    def rated_kw(self) -> float:
        """Trafonun kW cinsinden anma gucu."""
        return self.transformer_kva * self.power_factor

    def socket_list(self) -> List[float]:
        """Soket guc siniflarini (kW) buyukten kucuge dondurur."""
        return (
            [200.0] * self.n_socket_200
            + [180.0] * self.n_socket_180
            + [120.0] * self.n_socket_120
        )

    @property
    def n_sockets(self) -> int:
        return self.n_socket_120 + self.n_socket_180 + self.n_socket_200

    @property
    def installed_kw(self) -> float:
        """Toplam kurulu istasyon (soket) gucu (kW)."""
        return float(sum(self.socket_list()))

    @property
    def diversified_demand_kw(self) -> float:
        """Cesitlilik faktorlu (esZamanli) istasyon talebi (kW) - madde 6/9."""
        return self.installed_kw * self.diversity_factor

    @property
    def ramp_kw_per_min(self) -> float:
        """
        Rampa hizi (kW/dakika) - madde 4. Kontrol edilebilir kurulu sarj gucunun
        ramp_frac_per_min yuzdesi; cok kucuk kurulumlar icin ramp_floor_kw_per_min
        alt tabani uygulanir. Mutlak sabit yerine olceklenebilir tanim.
        """
        return max(self.ramp_floor_kw_per_min, self.installed_kw * self.ramp_frac_per_min)


# --------------------------------------------------------------------------- #
# Cok amacli optimizasyon agirliklari (alpha, beta, gamma)
# --------------------------------------------------------------------------- #
@dataclass
class Weights:
    """
    Multi-objective agirliklar (dashboard slider'lari ile ayarlanir):
        alpha -> Sarj Suresi (hizli bitir)
        beta  -> SOH Koruma  (dusuk C-rate / bataryayi koru)
        gamma -> Maliyet     (ucuz fiyat/dilimden yararlan, puanti tirasla)
    """
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5


# --------------------------------------------------------------------------- #
# Fiyatlandirma (PTF/SMF veya 3-zamanli tarife)
# --------------------------------------------------------------------------- #
@dataclass
class PricingConfig:
    """
    Iki olcek:
      - mode="PTF" : EPIAS PTF/SMF piyasa fiyati (buyuk fabrikalar).
      - mode="TARIFE": 3-zamanli sanayi tarifesi (daha kucuk tesisler).
    """
    mode: str = "PTF"   # "PTF" veya "TARIFE"

    # ---- EPIAS canli PTF cekimi (opsiyonel) ----
    # use_epias=True ve gecerli kullanici adi/sifre verilirse PTF, EPIAS
    # Seffaflik Platformu API'sinden cekilir; aksi halde sentetik egri kullanilir.
    use_epias: bool = False
    epias_username: str = ""
    epias_password: str = ""
    # Veri cekilecek bitis tarihi (None -> dun). Son `days` gun cekilir.
    data_end_date: str = ""   # "YYYY-MM-DD" veya bos

    # ---- 3-zamanli tarife (TL/kWh) - dagitim+enerji dahil ----
    price_night: float = 2.20   # Gece  22:00-06:00
    price_day: float = 3.80     # Gunduz 06:00-17:00
    price_peak: float = 6.50    # Puant 17:00-22:00
    night_start_hour: int = 22
    day_start_hour: int = 6
    peak_start_hour: int = 17

    # ---- PTF/SMF piyasa parametreleri (2026 EPIAS gerceklerine kalibre, TL/MWh) ----
    # 2026 GERCEK PROFILI (EPIAS Seffaflik, ornek 07.06.2026):
    #   - GUNDUZ 08:00-16:00: gunes bollugunda PTF GENIS bir pencerede ~0'a iner
    #     (gercekte ~10 saatlik dar OLMAYAN bir bant; eski model burada hataliydi).
    #   - AKSAM 19:00-21:00: puant tepe azami fiyata (~2700) firlar.
    #   - GECE/SABAH: ~baz seviye (~700-800), yani gunduz sifirindan PAHALIDIR.
    #   - Gunluk ortalama mevsim/havaya gore cok oynar (Mart'26 ~1620, Haz'26 ~390).
    ptf_night_base: float = 800.0     # gece/baz seviye (TL/MWh)
    ptf_night_dip: float = 250.0      # gece yarisi (03-05) ek dip
    ptf_morning_bump: float = 350.0   # sabah omuz (07-09)
    ptf_evening_peak: float = 2700.0  # aksam puant tepe (azami fiyat civari)
    ptf_cap: float = 3000.0           # 2026 piyasa azami fiyati (TL/MWh)
    ptf_floor: float = 0.0            # taban: yuksek-solar gunduzde ~0 olabilir
    ptf_solar_min: float = 0.20       # dusuk-solar (bulutlu/kis) gun bastirma
    ptf_solar_max: float = 1.00       # yuksek-solar gun -> gunduz penceresi ~0
    smf_spread: float = 350.0         # SMF'nin PTF'den sapma genligi (dengesizlik)

    # AYLIK GUNES (SOLAR) FAKTORU (N4 - mevsimsel durustluk): gunduz PTF'sinin
    # ~0'a inme derinligi mevsime baglidir. Ankara aylik kuresel isinim (GHI)
    # mertebesine gore normalize edilmis carpan, Ocak..Aralik. Hem sentetik PTF
    # uretiminde (solar_depth x ay faktoru) hem de enerji tasarrufunun yillik
    # projeksiyon mevsim duzeltmesinde (energy_seasonal_factor) kullanilir.
    monthly_solar_factor: tuple = (
        0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00, 0.95, 0.80, 0.58, 0.38, 0.27,
    )


# --------------------------------------------------------------------------- #
# Trafo Termal Omur modeli (IEC 60076-7:2018) - madde 2
# --------------------------------------------------------------------------- #
@dataclass
class ThermalConfig:
    """
    IEC 60076-7:2018 "Loading guide for oil-immersed power transformers" sicak-nokta
    (hot-spot) sicakligi ve omur tuketimi modeli (fark-denklemi cozumu, madde 8.2.2).

    SICAK-NOKTA (FARK DENKLEMI) MODELI:
      K(t)        = (baz+sarj)/S_anma                              (p.u. yuklenme)
      Δθo,ult     = Δθor · ((1 + R·K²)/(1 + R))^x                  (ust-yag nihai artisi)
      Δθo[t]      = Δθo[t-1] + (Δt/(k11·τo))·(Δθo,ult − Δθo[t-1])  (ust-yag, yag ataleti)
      Δθh1[t]     = Δθh1[t-1] + (Δt/(k22·τw))·(k21·Δθhr·K^y − Δθh1[t-1])
      Δθh2[t]     = Δθh2[t-1] + (Δt/((1/k22)·τo))·((k21−1)·Δθhr·K^y − Δθh2[t-1])
      Δθh        = Δθh1 − Δθh2                                     (sicak-nokta gradyani)
      θh[t]       = θa(t) + Δθo[t] + Δθh[t]                        (sicak-nokta sicakligi)

    BAGIL YASLANMA HIZI (madde 6.3, normal/termal-iyilestirilMEMIS kraft kagit, 98°C ref):
      V(t)        = 2^((θh(t) − 98)/6)
      Tuketilen omur (saat) = Σ V(t)·Δt  (98°C referansta V=1, "normal" omur hizi)

    Parametreler IEC 60076-7:2018 Tablo (orta/buyuk guc trafosu, ONAN) onerilen
    degerleridir.
    """
    # ---- IEC fark-denklemi termal model sabitleri (orta guc ONAN) ----
    x_oil_exp: float = 0.8        # yag ustsel (oil exponent)
    y_wind_exp: float = 1.3       # sargi ustsel (winding exponent)
    R_ratio: float = 6.0          # yuk kaybi / bos calisma kaybi orani
    k11: float = 0.5              # termal model sabiti
    k21: float = 2.0              # termal model sabiti
    k22: float = 2.0              # termal model sabiti
    tau_o_min: float = 210.0      # ortalama yag zaman sabiti (dakika)
    tau_w_min: float = 10.0       # sargi (hot-spot) zaman sabiti (dakika)

    # ---- Anma artislari (rated rises) ----
    # Anma yukunde (K=1) ust-yag artisi ve sicak-nokta-ustyag gradyani. Toplam
    # 52 + 26 = 78 K sicak-nokta artisi; 20°C ortamda θh,anma = 98°C (IEC normal
    # yalitim referans tasarimi -> anmada V=1).
    dtheta_or: float = 52.0       # Δθor: anma ust-yag artisi (K)
    dtheta_hr: float = 26.0       # Δθhr = H·gr: anma sicak-nokta-ustyag gradyani (K)

    # ---- Bagil yaslanma / omur ----
    hs_reference_c: float = 98.0          # IEC normal kagit referansi (V=1)
    aging_base: float = 2.0               # V = 2^((θh−98)/6)
    aging_doubling_k: float = 6.0         # her +6°C'de yaslanma ikiye katlanir

    # TRAFO NORMAL/TASARIM OMRU = 30 YIL (madde 2 - kullanici varsayimi):
    # Trafonun omru, REFERANS yaslanma hizinda (V=1, sicak-nokta 98°C) 30 yil
    # ALINIR. Yani toplam omur butcesi = 30 × 8760 = 262.800 esdeger-saattir.
    # Tuketilen omur yuzdesi DAIMA bu 30 yillik butceye gore ifade edilir; boylece
    # "30 yilda tuketilen omur" ve "bu donemde tuketilen omur" dogrudan 30 yila
    # oranlanir. (Eski 180.000 saat/98°C IEC normal-omur referansi yerine.)
    normal_life_hours: float = 262800.0   # = 30 yil × 8760 saat (30 yil referans omru)

    # FIZIKSEL TASARIM OMRU TAVANI (madde 2):
    # Trafonun fiziksel/tasarim omru 30 yildir (nem/busing/OLTC/mekanik). Termal
    # esdeger omur bundan uzun cikabilir; gosterimde 30 yil tavaniyla kirpilir.
    design_life_years: float = 30.0

    # 30-YILLIK OMUR TUKETIMI EKSTRAPOLASYONU (madde 3):
    # Simulasyon EN KOTU 100 gunu (Mayis-Agustos, Ankara) kapsar. 30 yillik omur
    # tuketimi, bu pencerenin yaslanma hizi MEVSIMSEL DUZELTME ile yila tasinir
    # (kis aylarinda θa dusuk -> V ustel olarak cok kuculur). Yaklasim:
    #   yillik_LoL ≈ pencere_LoL × (365/sim_gun) × mevsim_faktoru
    #   mevsim_faktoru = <2^(θa_gun/6)>_yil / <2^(θa_gun/6)>_pencere   (<1)
    # mevsim_faktoru, Ankara aylik ortalama sicakliklarindan hesaplanir.
    extrapolation_years: float = 30.0

    # ---- Gercek ANKARA ortam sicakligi (madde 1) ----
    # Simulasyon EN KOTU senaryo icin MAYIS basindan baslar (1 numarali ay=Ocak).
    sim_start_month: int = 5      # Mayis
    sim_start_day: int = 1
    # Ankara aylik ORTALAMA gunluk sicaklik (°C) ve gunluk salinim genligi (yari
    # tepe-dip), Ocak..Aralik. (Kaynak: iklim normalleri, climate-data.org /
    # WeatherSpark.) Termal model bu serileri gun-ici sinuzoidal profile yayar.
    ankara_monthly_mean_c: tuple = (
        0.5, 2.0, 6.0, 11.0, 15.7, 20.0, 23.5, 23.5, 18.5, 12.5, 6.5, 2.0,
    )
    ankara_monthly_amp_c: tuple = (
        5.0, 6.0, 7.0, 7.0, 7.0, 7.2, 7.5, 7.5, 7.5, 7.0, 6.0, 5.0,
    )
    ambient_daily_noise_c: float = 1.5    # gunluk hava sapmasi (deterministik)

    # Trafo yenileme maliyeti (1600 kVA OG trafo, montaj dahil, TL)
    transformer_cost_tl: float = 4_000_000.0


# --------------------------------------------------------------------------- #
# Batarya / SOH ve genel finansal parametreler
# --------------------------------------------------------------------------- #
@dataclass
class FinancialConfig:
    """SOH, demand charge ve genel finansal parametreler (TL)."""

    # GUC / DEMAND CHARGE ve GUC ASIM CEZASI (EPDK mevzuati)
    # ------------------------------------------------------------------
    # Sozlesme gucu (kW): dagitim sirketiyle anlasilan azami cekis gucu.
    # Trafo etkin anmasinin (1520 kW @ cosφ=0.95) altinda; ekonomik tepe esigi.
    contracted_demand_kw: float = 1300.0
    # Aylik guc/demand bedeli (TL/kW/ay).
    demand_charge_tl_per_kw: float = 90.0
    # GUC BEDELI REJIMI (N2 - mevzuat duzeltmesi):
    #   "EPDK"   -> guc bedeli SOZLESME GUCU uzerinden SABIT tahakkuk eder (TR
    #               uygulamasi). Tepe dususu yalnizca (i) ASIM CEZASINI azaltir ve
    #               (ii) "daha dusuk sozlesme gucu secebilme" firsatini yaratir;
    #               olculen tepe dustugu icin guc bedeli kendiliginden DUSMEZ.
    #   "DEMAND" -> ABD-tarzi: olculen aylik tepe x birim bedel (eski davranis;
    #               karsilastirma/duyarlilik icin secilebilir).
    billing_mode: str = "EPDK"
    # TEPE OLCUM PERIYODU (N3): EPDK/OSOS sayaclari tepe gucu 15 DAKIKALIK
    # ORTALAMA uzerinden olcer; dakikalik anlik tepe kullanmak cezayi/tasarrufu
    # sistematik sisirir. Demand/ceza hesabi bu periyodun ortalamasiyla yapilir.
    demand_interval_min: int = 15
    # GUC ASIM BEDELI: EPDK "Elektrik Piyasasi Tarifeler Yonetmeligi" uyarinca
    # sozlesme gucunu asan abone, asilan guc icin guc bedelinin KATI tutarinda
    # ceza oder. Tipik uygulama, asim donemi icin guc bedelinin ~3 katidir
    # (kesin oran dagitim sirketi/EPDK onayli tarifeye gore degisir).
    demand_penalty_multiplier: float = 3.0
    # Senaryo-7 icin baz yuk artis orani (baz +%15 karsilastirmasi).
    base_increase_pct: float = 0.15

    # USD/TRY KURU (madde 2): trafo ve diger maliyetleri DOLAR cinsinden de
    # gostermek icin. Trafo ekipmani genelde doviz endeksli fiyatlanir; bu kur
    # ile TL maliyetler dolara cevrilir (guncel piyasa kuruna gore degistirilebilir).
    # Guncelleme: 11.06.2026 piyasa kuru ~46.15 TL/USD.
    usd_try_rate: float = 46.15

    @property
    def demand_penalty_tl_per_kw(self) -> float:
        """Asilan kW basina ceza = guc bedeli x ceza kati (EPDK yaklasimi)."""
        return self.demand_charge_tl_per_kw * self.demand_penalty_multiplier

    # Batarya / SOH
    battery_cost_tl_per_kwh: float = 4500.0  # pak maliyeti
    cycle_life: float = 1500.0               # %20 kayba kadar tam cevrim
    eol_capacity_loss: float = 0.20          # omur sonu kayip orani
    soh_crate_k: float = 0.6                 # C-rate stres katsayisi

    @property
    def soh_loss_per_stress_kwh_frac(self) -> float:
        """
        Stres-kWh basina SOH kayip ORANI (birimsiz):
            = eol_capacity_loss / (cycle_life)   ... kapasiteye normalize edilince
        Asagidaki kullanim: loss_frac = stress_throughput / (capacity*cycle_life) * eol_loss
        (Burada per-vehicle hesaplanir; bu property referans amaclidir.)
        """
        return self.eol_capacity_loss / self.cycle_life


# --------------------------------------------------------------------------- #
# Senaryo (B2B) yapilandirmasi
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioConfig:
    """
    B2B senaryolari:
      - AVM   : yuksek gunduz baz yuk + Maksimum Gecikme Limiti (delay cap) ZORUNLU.
      - FABRIKA (Lojistik Filo): gece bekleyebilir, gecikme limiti YOK ->
                ucuz gece/dusuk-fiyat saatlerinden sonuna kadar yararlanir.
    """
    name: str = "FABRIKA"   # "AVM" veya "FABRIKA"

    # Kalici arac filosu buyuklugu (kumulatif SOH analizi icin sabit kimlikler).
    # Filo, soketleri kuyrukla doyuracak kadar buyuk tutulur; boylece algoritmasiz
    # stratejisi yuksek baz-yuk saatlerinde trafoyu SUREKLI asiri yukler (overload)
    # ve termal yaslanma farki olculebilir hale gelir.
    fleet_size_avm: int = 10         # DEFAULT: 10 arac
    fleet_size_factory: int = 10     # DEFAULT: 10 arac (depo/filo)

    # Gunluk sarj olma olasiligi (her arac icin).
    daily_charge_prob_avm: float = 0.80
    daily_charge_prob_factory: float = 0.95

    # AVM icin maksimum kalma/gecikme limiti (dakika) - Max Delay Cap.
    avm_max_stay_min: int = 240

    # Dashboard'dan elle ayarlanan arac sayisi (0 -> senaryo varsayilani kullanilir)
    fleet_size_override: int = 0

    # CIKIS SAATI BELIRSIZLIGI (N7 - operasyonel gerceklik): gercek sahada
    # optimizer aracin cikis saatini TAM bilemez (kullanici beyani/tahmin).
    # Planlama, tahmini cikis = gercek cikis + N(0, sigma) ile yapilir (dk).
    # 0 = tam bilgi (gun-oncesi kesin beyan varsayimi). Fiziksel cikis daima
    # GERCEK saattir; tahmin hatasi tamamlanma oranini dusurebilir (raporlanir).
    dep_uncertainty_min: float = 0.0

    @property
    def is_factory(self) -> bool:
        n = self.name.upper()
        return n.startswith("FAB") or n.startswith("FIL") or n.startswith("FLEET")

    def fleet_size(self) -> int:
        """Aktif filo buyuklugu (override varsa onu, yoksa senaryo varsayilanini)."""
        if self.fleet_size_override and self.fleet_size_override > 0:
            return self.fleet_size_override
        return self.fleet_size_factory if self.is_factory else self.fleet_size_avm

    def daily_charge_prob(self) -> float:
        return self.daily_charge_prob_factory if self.is_factory else self.daily_charge_prob_avm


@dataclass
class SimConfig:
    """Tum alt-yapilandirmalari toplayan ust seviye kapsayici."""
    days: int = DAYS_DEFAULT
    seed: int = 42
    station: StationConfig = field(default_factory=StationConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    financial: FinancialConfig = field(default_factory=FinancialConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    weights: Weights = field(default_factory=Weights)

    # ALGORITMA DEVREYE GIRME POLITIKASI (madde 6):
    #   "always"    -> algoritma her dakika calisir.
    #   "peak_only" -> algoritma SADECE trafo doluluk orani (baz_yuk/anma)
    #                  peak_loading_threshold'u astiginda devreye girer; aksi
    #                  halde sistem algoritmasiz (yonetimsiz, tam guc) gibi davranir.
    activation_mode: str = "always"
    peak_loading_threshold: float = 0.60   # %60 trafo doluluk -> "puant"

    @property
    def total_minutes(self) -> int:
        return self.days * MINUTES_PER_DAY
