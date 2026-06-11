# Finansal Model — Formüller, Varsayımlar ve Kaynaklar (Yatırımcı Notu)

Bu belge, dashboard'daki tüm finansal KPI'ların **hangi formüllere** dayandığını,
**hangi varsayımları** kullandığını ve bunların **gerçek dünya kaynaklarını** açıklar.
Tüm parametreler `src/config.py` içinde tek yerde toplanmıştır ve değiştirilebilir.

> Not: Birim fiyatlar (PTF, dağıtım/güç bedeli, batarya, trafo) **tarih ve
> dağıtım şirketine göre değişir**. Modelde varsayılan değerler 2024–2025 Türkiye
> koşullarına yakın seçilmiştir; kesin teklif için ilgili EPDK onaylı tarife ve
> güncel piyasa fiyatları girilmelidir. PTF, **EPİAŞ Şeffaflık Platformu API**'sinden
> canlı çekilebilir (`src/epias.py`).

---

## 1. Enerji Maliyeti

Her dakikada çekilen şarj enerjisinin, o dakikadaki birim fiyatla çarpımı:

```
E_dakika(t) = P_şarj(t) [kW] × (1/60) [saat]        → kWh
Enerji Maliyeti = Σ_t  E_dakika(t) × fiyat(t) [TL/kWh]
```

- **PTF ölçeği (büyük tüketici):** `fiyat(t) = PTF(t) / 1000` (TL/MWh → TL/kWh).
  PTF, EPİAŞ Gün Öncesi Piyasası **MCP (Piyasa Takas Fiyatı)** verisidir.
- **3-Zamanlı Tarife ölçeği (küçük tesis):** EPDK onaylı sanayi çok-zamanlı tarife:
  Gece / Gündüz / Puant (TL/kWh).

**Tasarruf** = `Enerji Maliyeti(Algoritmasız) − Enerji Maliyeti(Algoritmalı)`.

**Yıllık projeksiyon (N4 — mevsim düzeltmesi):** Simülasyon penceresi yaz ağırlıklıdır;
gündüz "solar ~0" ucuz penceresi kışın sığlaşır. Bu yüzden yıllık enerji tasarrufu düz
`×365/gün` ile değil, **mevsim faktörüyle** taşınır:

```
proxy(ay)         = 0.5 + 0.5 × solar_f(ay)        (yarısı akşam-puant kaçınması: yıl boyu sabit;
                                                     yarısı ucuz-solar pencere derinliği: aylık)
enerji_mevsim_f   = <proxy>_yıl / <proxy>_pencere   (yaz penceresi için < 1)
Yıllık Enerji Tasarrufu = Dönem Tasarrufu × (365/gün) × enerji_mevsim_f
```

`solar_f(ay)`: Ankara aylık güneşlenme (GHI) mertebesine göre normalize çarpan
(`monthly_solar_factor`, config). Aynı tablo sentetik PTF üretiminde de kullanılır;
12 aylık simülasyonlar gerçek mevsimselliği doğrudan taşır.

---

## 2. Güç Bedeli ve Güç Aşım Cezası (N2, N3 — EPDK rejimi)

**Tepe ölçümü (N3):** EPDK/OSOS sayaçları tepe gücü **15 dakikalık ORTALAMA**
üzerinden ölçer (`demand_interval_min = 15`). Model dakikalık anlık tepeyi değil,
15-dk ortalamayı faturalama/ceza tabanı olarak kullanır (anlık tepe trafo fiziği
için ayrıca raporlanır). Dakikalık sivri tepeler ortalamada törpülenir; bu,
cezayı/tasarrufu sistematik şişirmeyi önler.

**Güç bedeli rejimi (N2, `billing_mode`):**

```
Aylık 15-dk Tepe  P15,ay = max( ort_15dk(Baz+Şarj) )           [kW]  (her 30 günde)

EPDK (varsayılan, TR mevzuatı):
  Güç Bedeli      = Σ_ay  SÖZLEŞME_GÜCÜ × demand_charge        (SABİT; tepeden bağımsız)
  Güç Aşım Cezası = Σ_ay  max(0, P15,ay − Sözleşme) × (demand_charge × ceza_katı)

DEMAND (ABD-tarzı, karşılaştırma için):
  Güç Bedeli      = Σ_ay  P15,ay × demand_charge
```

- EPDK rejiminde güç bedeli iki stratejide **aynıdır**; "demand tasarrufu"
  yalnızca **aşım cezası farkıdır**. Tepe düşüşünün ikinci parasal getirisi,
  **daha düşük sözleşme gücü seçebilme** fırsatıdır (dashboard'da ayrı kalemde
  parasallaştırılır: `tıraşlanan kW × bedel × 12`).
- `demand_charge` (90 TL/kW/ay) ve `ceza_katı` (3×) dağıtım şirketi tarifesine
  göre değişir; tek parametreyle ayarlanır.

**Algoritmanın faydası (EPDK rejiminde):** aşım cezasını sıfıra çeker ve sözleşme
gücünü düşük tutarak büyümeye alan açar; trafoyu aşırı yüklememenin parasal
karşılığı budur.

---

## 3. Trafo Termal Ömür (IEC 60076-7:2018) → Parasal Karşılık (Madde 1, 2, 3)

Trafonun yalıtım ömrü, **sıcak-nokta sıcaklığına** üstel bağlıdır. Model, **IEC
60076-7:2018** "Loading guide for oil-immersed power transformers" standardının
**fark-denklemi çözümünü** (madde 8.2.2) kullanır ve **gerçek Ankara ortam
sıcaklığını** (Mayıs–Ağustos, en kötü senaryo — madde 1) girdi alır.

```
K(t)     = (Baz+Şarj)(t) / S_anma                          (p.u. yüklenme)
Δθo,ult  = Δθor · ((1 + R·K²)/(1 + R))^x                    (üst-yağ nihai artışı)
Δθo[t]   = Δθo[t-1] + (Δt/(k11·τo))·(Δθo,ult − Δθo[t-1])    (yağ ataleti, τo)
Δθh1[t]  = Δθh1[t-1] + (Δt/(k22·τw))·(k21·Δθhr·K^y − Δθh1[t-1])
Δθh2[t]  = Δθh2[t-1] + (Δt·k22/τo)·((k21−1)·Δθhr·K^y − Δθh2[t-1])
Δθh      = Δθh1 − Δθh2                                      (sıcak-nokta gradyanı)
θh(t)    = θa(t) + Δθo(t) + Δθh(t)                          (sıcak-nokta °C; θa = Ankara)

V(t)     = 2^((θh(t) − 98)/6)        (bağıl yaşlanma; normal kâğıt, 98°C'de V=1)
Tüketilen Ömür (saat) = Σ_t V(t) · (1/60)
```

**IEC parametreleri** (Tablo, orta güç ONAN): x=0.8, y=1.3, R=6, k11=0.5, k21=2.0,
k22=2.0, τo=210 dk, τw=10 dk. Anma artışları Δθor=52 K, Δθhr=26 K → 20°C ortamda
θh,anma=98°C (IEC normal yalıtım tasarımı, V=1).

### 30 Yıllık Ömür Ekstrapolasyonu ve Ertelenen Değişim Maliyeti (Madde 3)

Simülasyon **en kötü 100 günü** (yaz) kapsar. Tipik bir trafo ömrü (30 yıl) boyunca
tüketilen ömür, **mevsimsel düzeltme** ile tahmin edilir (kış aylarında θa düşük →
V üstel olarak küçülür):

```
mevsim_faktörü s = <2^(θa_gün/6)>_yıl / <2^(θa_gün/6)>_pencere        (< 1; Ankara normalleri)
yıllık_LoL       = (pencere_LoL / sim_gün) × 365 × s
Termal-Eşdeğer Ömür (yıl) = normal_ömür_saat / yıllık_LoL
30-yıl tüketilen kesir    = yıllık_LoL × 30 / normal_ömür_saat
```

**DLM ömür uzaması:** `Termal-Eşdeğer Ömür(Algoritmalı) − (Algoritmasız)`.

**Ertelenen Trafo Değişim Maliyeti (KPI):**
```
Ertelenen Değişim = (30-yıl kesir_Algoritmasız − 30-yıl kesir_Algoritmalı) × Trafo_Yenileme_Maliyeti
```

- `normal_ömür_saat = 262.800 saat` (= 30 yıl × 8760; V=1 referans hızında 30 yıllık
  ömür bütçesi — config `normal_life_hours`).
- `Trafo_Yenileme_Maliyeti` (varsayılan **4.000.000 TL**, 1600 kVA OG trafo + montaj).
- **Fiziksel tasarım ömrü tavanı = 30 yıl** (nem/busing/OLTC/mekanik).
- Yıllık projeksiyonda termal kalem `seasonal_aging_factor` ile (kış ortamı serin →
  V üstel küçülür), enerji kalemi `energy_seasonal_factor` ile düzeltilir (N4).

> Not: Doğru boyutlandırılmış (madde 9, overload'suz) bu sistemde sıcak-nokta IEC
> referansının (98°C) genelde altındadır; termal yaşlanma çok düşüktür ve trafo ömrü
> **fiziksel tasarım ömrüyle** sınırlıdır. DLM termal omru daha da uzatır; parasal
> karşılık 30 yıllık ufukta korunan ömür kesridir (mütevazı ama gerçek).

---

## 4. Batarya Sağlığı (SOH) → Parasal Karşılık (Madde 8)

Batarya kapasite kaybı, **işlenen enerjiye** ve **C-rate stresine** bağlıdır.

```
C-rate(t)    = P(t) / Kapasite
stres_kWh    = Σ E_dakika · (1 + k · C-rate²)            (k = soh_crate_k = 0.6)
SOH_kaybı(%) = stres_kWh / (Kapasite · çevrim_ömrü) × eol_kayıp × 100
```

- `çevrim_ömrü = 1500` tam çevrim, `eol_kayıp = %20` (ömür sonu kabul edilen kayıp).
- Yüksek C-rate (algoritmasiz, hızlı şarj) stresi **kuadratik** artırır → daha hızlı yıpranma.

**Parasal fayda (Madde 8):** Korunan SOH, **batarya değişiminin ertelenmesi**dir:

```
Paket Değeri      = Kapasite [kWh] × batarya_maliyeti [TL/kWh]    (varsayılan 4500 TL/kWh)
Korunan Bedel(araç) = (SOH_kaybı_algoritmasız − SOH_kaybı_algoritmalı) × Paket Değeri
Geciktirilen Değişim Bedeli = Σ_araç  Korunan Bedel(araç)
```

> **ROI ataması (N5):** Batarya kimin malıysa fayda onundur. **FABRIKA/filo**
> senaryosunda araçlar tesise aittir → SOH kazancı operatör ROI toplamına **girer**.
> **AVM** senaryosunda bataryalar müşterinindir → operatör toplamına **girmez**;
> "müşteri faydası" olarak ayrı raporlanır.
>
> **Kalibrasyon notu:** C-rate stres katsayısı `k = 0.6` literatür kalibrasyonu
> yapılmamış bir yaklaşımdır; kalendar yaşlanma, hücre sıcaklığı ve SoC penceresi
> etkileri modelde yoktur. SOH kalemi **gösterge niteliğinde** okunmalıdır
> (yol haritası Faz 3: NREL/Wang tipi yayınlanmış yaşlanma modeline kalibrasyon).

---

## 5. Power-Shaving (Rezerv Yük) Yatırım Getirisi

```
Tıraşlama (kW)   = Tepe(Algoritmasız) − Tepe(Algoritmalı)
Tıraşlama (%)    = Tıraşlama / Tepe(Algoritmasız) × 100
İlave İstasyon   = Tıraşlama / ortalama_istasyon_gücü
```

Yorum (N6 — modelle hizalı anlatım): Yaratılan headroom, **yeni talebin de DLM tavanı
altında yönetilmesi koşuluyla** ilave DC istasyon / EV / baz yük büyümesine alan açar.
Doğru boyutlandırılmış (overload'suz) bu sistemde **termal ömür kazancı küçüktür**;
trafo-tarafı parasal değer, **aşım cezalarının önlenmesi** ve **sözleşme gücünün düşük
tutulabilmesidir**. "Trafo değişimi ertelendi" iddiası yalnızca gerçek overload'lu
senaryolar (ör. baz yük +%15, olay-tabanlı naive) için kurulabilir.

---

## 6. Senaryo: Baz Yük +%15 (Madde 7)

Aynı gün, **aynı istasyon ve araç koşulları** sabitken baz yük %15 artırılır.
Amaç: mevcut istasyon sayısının, baz yük arttığında **güç aşım cezasına** yol
açtığını göstermek.

```
Tepe(+%15) = max_t ( 1.15 · Baz Yük(t) + Şarj(t) )
Ek Ceza    = Güç Aşım Cezası(+%15) − Güç Aşım Cezası(mevcut)
```

Ceza fiyatı Bölüm 2'deki EPDK güç aşım yaklaşımıyla (güç bedeli × ceza katı)
hesaplanır.

---

## 7. Çeşitlilik Faktörü, Güç Faktörü, Şarj Verimi ve Ramp (Madde 4, 6, 7, 9)

**Güç faktörü (Madde 6):** Trafo etkin (aktif) gücü = `kVA × cosφ`. Tesis ortalama
güç faktörü **cosφ = 0.95** alınır (modern DC şarjda aktif PFC ~0.98–0.99; tesis
ortalaması ~0.95; EPDK reaktif ceza eşiği cosφ≥0.90'ın üstünde). → `1600 × 0.95 =
1520 kW` etkin anma.

**Çeşitlilik / eşzamanlılık faktörü (Madde 6, 9):** IEC 60364-7-722, bir Yük Yönetim
Sistemi (LMS) **yoksa** tasarım çeşitlilik faktörünü 1 alır. Pratik **talep
tahmininde** bir DC şarj kümesinin gerçek eşzamanlı talebi kuruludan düşüktür.
Modelde:
```
Algoritma-öncesi (naive, LMS yok) eşzamanlı şebeke talebi = çeşitlilik × kurulu_güç
Algoritma (optimize) = IEC 60364-7-722'deki LMS — talebi trafo headroom'unda yönetir
```
Varsayılan `çeşitlilik = 0.60`. **Boyutlandırma kuralı (Madde 9):** istasyon sayısı,
çeşitlilikli talep trafo anmasının **%20–30**'unda olacak şekilde seçilir
(2×200 + 1×180 + 1×120 = 700 kW → 700×0.60 = 420 kW ≈ %27.6). Baz tepe (%60) +
çeşitlilikli istasyon (%27.6) = %87.6 < %100 → **algoritma öncesinde bile overload
yoktur.**

**Şarj verimi (Madde 7):** DC hızlı şarj ortalama verimi **%93** (~%92–95).
Şebekeden çekilen (faturalanan + trafo yükü) güç = batarya gücü / verim.

**Ramp hızı (Madde 4):** EV şarj cihazları için "kW/dk" cinsinden **zorunlu bir
standart yoktur**; cihaz kendisi ISO 15118 / IEC 61851 ile saniyeler içinde rampa
yapar. Sınır, **saha EMS'inin güç-kalitesi** (gerilim dalgalanması / flicker —
IEC 61000-3-3 / IEC 61000-3-11) için koyduğu bir yumuşatma tercihidir. Bu nedenle
ramp, kontrol edilebilir kurulu gücün bir **yüzdesi** olarak ifade edilir:
`ramp_kW/dk = ramp_frac × kurulu_güç` (varsayılan **%10/dk**, alt taban 30 kW/dk).

---

## Varsayılan Parametre Özeti

| Parametre | Varsayılan | Kaynak/Gerekçe |
|---|---|---|
| Trafo anma | 1600 kVA / 1520 kW | Donanım; etkin kW = kVA × cosφ |
| Güç faktörü cosφ | 0.95 | Tesis ortalaması (madde 6) |
| Baz yük tepe | %60 (912 kW) | Tasarım (madde 9) |
| Çeşitlilik faktörü | 0.60 | IEC 60364-7-722 talep tahmini (madde 6) |
| Çeşitlilikli talep | %20–30 trafo | Boyutlandırma hedefi (madde 9) |
| Şarj verimi | %93 | DC hızlı şarj ortalaması (madde 7) |
| Ramp | %10/dk (kurulu güç) | Güç kalitesi/EMS, IEC 61000-3-3/-11 (madde 4) |
| Güç bedeli | 90 TL/kW/ay | EPDK güç bedeli mertebesi |
| Güç bedeli rejimi | EPDK (sözleşme bazlı sabit) | N2; "DEMAND" (tepe bazlı) seçilebilir |
| Tepe ölçüm periyodu | 15 dk ortalama | N3; EPDK/OSOS ölçüm pratiği |
| Güç aşım ceza katı | 3× | EPDK aşım uygulaması (tarifeye göre değişir) |
| Sözleşme gücü | 1300 kW | Tesis sözleşmesi (< etkin anma) |
| Naive talep modeli | diversity (tavan) / event (üst-sınır) | N1; bant olarak raporlanır |
| Çıkış saati belirsizliği | ±0 dk (ayarlanabilir) | N7; >0 ile saha gerçekliği testi |
| Aylık solar faktörü | Ankara GHI normalize | N4; PTF mevsimselliği + enerji yıllık düzeltme |
| Batarya maliyeti | 4500 TL/kWh | 2024–25 pak fiyat mertebesi |
| Çevrim ömrü / EOL | 1500 / %20 | Li-ion tipik |
| Trafo yenileme | 4.000.000 TL | 1600 kVA OG trafo + montaj |
| Termal model | IEC 60076-7:2018 | Fark-denklemi, ONAN orta güç (madde 2) |
| Ortam sıcaklığı | Gerçek Ankara (Mayıs–Ağu) | İklim normalleri (madde 1) |
| Normal ömür bütçesi | 262.800 saat (30 yıl, 98°C ref.) | V=1 hızında 30 yıl (config) |
| Tasarım ömrü tavanı | 30 yıl | Fiziksel (nem/busing/OLTC) |
| PTF | EPİAŞ MCP (canlı) | EPİAŞ Şeffaflık Platformu |
