# -*- coding: utf-8 -*-
"""
config.py
=========
Tum simulasyon icin paylasilan sabitler ve yapilandirma nesneleri.

Kapsanan konular:
  - Donanim: 1600 kVA ana trafo, karisik DC soket parki (200/180/120 kW).
  - Baz yuk: trafo nominalinin ~%70'ine (≈1120 kW) ulasan tesis profili.
  - Fiyatlandirma: (a) PTF/SMF (EPIAS piyasa) - buyuk fabrika olcegi,
                   (b) 3-zamanli sanayi tarifesi - daha kucuk tesis olcegi.
  - Optimizasyon kisitlari: ±60 kW ramp, %80 SoC bitis, C-rate (SOH) tavani.
  - Trafo Termal Omur modeli (IEEE C57.91 benzeri) parametreleri.
  - Batarya SOH ve finansal (ROI) parametreleri.

Tum degerler gercek dunya verilerine yakin secilmistir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Zaman ekseni
# --------------------------------------------------------------------------- #
MINUTES_PER_DAY: int = 1440           # Simulasyon dakika bazli calisir.
HOURS_PER_MINUTE: float = 1.0 / 60.0  # dt = 1/60 saat (kW -> kWh donusumu)
DAYS_DEFAULT: int = 100               # 100 gun x 1440 dakika


# --------------------------------------------------------------------------- #
# Donanim (1600 kVA trafo + karisik DC soket parki)
# --------------------------------------------------------------------------- #
@dataclass
class StationConfig:
    """Sarj istasyonu donanim yapilandirmasi."""

    # Ana trafo: 1600 kVA. Basitlik icin guc faktoru 1.0 alinir (kVA ≈ kW),
    # boylece baz yuk zirvesi 0.70 * 1600 = 1120 kW olarak ifade edilir.
    transformer_kva: float = 1600.0
    power_factor: float = 1.0

    # Baz yuk, trafonun bu oranina kadar cikar (ispat icin yuksek tutulur).
    base_peak_frac: float = 0.70          # ≈ 1120 kW tepe

    # Optimize stratejinin trafoyu yuklemesine izin verdigi ust sinir (p.u.).
    # Optimize sarji, baz+sarj toplamini bu sinirin altinda tutar (peak-shaving).
    # Bodoslama (naive) bu siniri YOK sayar -> trafo asiri yuklenir (termal yaslanma).
    opt_max_loading_pu: float = 1.00      # optimize: trafoyu asma

    # KARISIK DC ALTYAPI: 1600 kVA'yi mantikli dolduran 6-8 soket.
    # KURULUM KURALI: toplam istasyon gucu, trafonun GECE headroom'unu
    # (rated - min(baz yuk)) gecmemelidir. Boylece tum istasyonlar gece ayni anda
    # tam guc calisabilir; GUNDUZ baz yuk yuksekken algoritma sarji kisarak trafoyu
    # korur. (Bodoslama kismadigi icin gunduz trafoyu asar -> overload.)
    n_socket_200: int = 2     # 2 x 200 kW
    n_socket_180: int = 3     # 3 x 180 kW
    n_socket_120: int = 2     # 2 x 120 kW   (toplam 7 soket, 1300 kW kurulu)

    # KESIN RAMPA HIZI KISITI:  |dP_total/dt| <= 60 kW / dakika.
    ramp_kw_per_min: float = 60.0

    # SARJ VERIMI (A4): sebekeden cekilen gucun bataryaya giren orani. DC hizli
    # sarjda AC/DC donusum + isil kayiplar ~%6-10'dur. Sebeke (trafo+fatura) gucu
    # = batarya gucu / verim. Yani fatura ve trafo yuku bataryaya gireninden buyuktur.
    charge_efficiency: float = 0.92

    # SABIT SARJ BITIS KURALI: tum araclar %80 SoC'de biter.
    target_soc: float = 0.80

    # MAKS SARJ UZATMA KATSAYISI (S) - madde 3:
    # Algoritma bir araci, tam-guc (en kotu senaryo / bodoslama) ile gereken
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


# --------------------------------------------------------------------------- #
# Trafo Termal Omur modeli (IEEE C57.91 benzeri)
# --------------------------------------------------------------------------- #
@dataclass
class ThermalConfig:
    """
    IEEE C57.91 benzeri sicak-nokta (hot-spot) sicakligi ve omur tuketimi.

    Buyukler:
      dtheta_to_rated : anma yukunde ust-yag sicaklik artisi (°C)
      dtheta_hs_rated : anma yukunde sicak-nokta'nin ust-yaga gore artisi (°C)
      R_ratio         : yuk kaybi / bos calisma kaybi orani
      n_exp, m_exp    : yag ve sargi ustelleri (ONAN icin ~0.8)
      tau_to_min      : ust-yag termal zaman sabiti (dakika)
      hs_reference_c  : referans sicak-nokta (110°C termal-iyilestirilmis kagit)
      normal_life_hours: referans sicaklikta normal yalitim omru (saat)
    """
    dtheta_to_rated: float = 55.0
    dtheta_hs_rated: float = 25.0    # 55 + 25 = 80°C sicak-nokta artisi (anma)
    R_ratio: float = 8.0
    n_exp: float = 0.8
    m_exp: float = 0.8
    tau_to_min: float = 180.0
    hs_reference_c: float = 110.0
    normal_life_hours: float = 180000.0   # ≈ 20.55 yil (termal yalitim referansi)

    # FIZIKSEL TASARIM OMRU TAVANI (madde 2):
    # Dusuk yuklenmede termal yaslanma ihmal edilebilir hale gelir ve IEEE LoL
    # modelinin "esdeger omru" yuzlerce/binlerce yila cikar (FIZIKSEL DEGIL).
    # Gercekte trafo omru nem, busing, OLTC, mekanik vb. nedenlerle tasarim
    # omruyle sinirlidir; esdeger omur bu tavanla kirpilir.
    design_life_years: float = 30.0

    # Ortam sicakligi (gun-ici sinuzoidal profil, °C)
    ambient_mean_c: float = 28.0
    ambient_amp_c: float = 7.0

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
    contracted_demand_kw: float = 1400.0
    # Aylik guc/demand bedeli (TL/kW/ay).
    demand_charge_tl_per_kw: float = 90.0
    # GUC ASIM BEDELI: EPDK "Elektrik Piyasasi Tarifeler Yonetmeligi" uyarinca
    # sozlesme gucunu asan abone, asilan guc icin guc bedelinin KATI tutarinda
    # ceza oder. Tipik uygulama, asim donemi icin guc bedelinin ~3 katidir
    # (kesin oran dagitim sirketi/EPDK onayli tarifeye gore degisir).
    demand_penalty_multiplier: float = 3.0
    # Senaryo-7 icin baz yuk artis orani (baz +%15 karsilastirmasi).
    base_increase_pct: float = 0.15

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
    # Filo, soketleri kuyrukla doyuracak kadar buyuk tutulur; boylece BODOSLAMA
    # stratejisi yuksek baz-yuk saatlerinde trafoyu SUREKLI asiri yukler (overload)
    # ve termal yaslanma farki olculebilir hale gelir.
    fleet_size_avm: int = 80         # duzenli musteri havuzu
    fleet_size_factory: int = 30     # depo/filo araclari

    # Gunluk sarj olma olasiligi (her arac icin).
    daily_charge_prob_avm: float = 0.80
    daily_charge_prob_factory: float = 0.95

    # AVM icin maksimum kalma/gecikme limiti (dakika) - Max Delay Cap.
    avm_max_stay_min: int = 240

    # Dashboard'dan elle ayarlanan arac sayisi (0 -> senaryo varsayilani kullanilir)
    fleet_size_override: int = 0

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
    #                  halde sistem bodoslama (yonetimsiz, tam guc) gibi davranir.
    activation_mode: str = "always"
    peak_loading_threshold: float = 0.60   # %60 trafo doluluk -> "puant"

    @property
    def total_minutes(self) -> int:
        return self.days * MINUTES_PER_DAY
