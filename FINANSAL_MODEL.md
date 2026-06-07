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

---

## 2. Demand Charge (Güç/Talep Bedeli) ve Güç Aşım Cezası

Sanayi aboneleri, **çektikleri tepe güç** üzerinden aylık güç bedeli öder ve
**sözleşme gücünü** aşarlarsa ceza öderler (EPDK *Elektrik Piyasası Tarifeler
Yönetmeliği*).

```
Aylık Tepe Güç P_peak,ay = max_t ( Baz Yük(t) + Şarj(t) )      [kW]   (her 30 günde)

Güç Bedeli       = Σ_ay  P_peak,ay × demand_charge        [TL/kW/ay]
Aşım (exceed)    = max(0, P_peak,ay − Sözleşme Gücü)      [kW]
Güç Aşım Cezası  = Σ_ay  exceed × (demand_charge × ceza_katı)
```

- `demand_charge` (varsayılan **90 TL/kW/ay**): bağlantı/güç bedeli.
- `ceza_katı` (varsayılan **3×**): EPDK uygulamasında sözleşme gücü aşımında,
  aşılan güç için güç bedelinin **katı** tutarında bedel tahakkuk eder. Kesin
  katsayı dağıtım şirketi tarifesine göre değişir; modelde tek parametre
  (`demand_penalty_multiplier`) ile ayarlanır.

**Algoritmanın faydası:** tepe gücü (peak-shaving ile) düşürerek hem güç bedelini
hem de aşım cezasını azaltır → doğrudan **trafoyu aşırı yüklememenin parasal
karşılığı** budur.

---

## 3. Trafo Termal Ömür (IEEE C57.91) → Parasal Karşılık (Madde 8)

Trafonun yalıtım ömrü, **sıcak-nokta sıcaklığına** üstel bağlıdır. Aşırı yüklenme
sıcak-noktayı yükseltir ve ömrü hızla tüketir.

```
K(t)     = (Baz+Şarj)(t) / S_anma                       (p.u. yüklenme)
ΔθTO_ult = ΔθTO,R · ((K²·R + 1)/(R + 1))^n              (üst-yağ nihai artışı)
ΔθTO(t)  : 1. derece gecikme (yağ ataleti, τ=180 dk)
ΔθH(t)   = ΔθH,R · K^(2m)                                (sıcak-nokta yağ-üstü artışı)
θH(t)    = θ_ortam(t) + ΔθTO(t) + ΔθH(t)                 (sıcak-nokta sıcaklığı, °C)

FAA(t)   = exp( 15000/383 − 15000/(θH(t)+273) )         (yaşlanma hızlandırma; 110°C'de =1)
Tüketilen Ömür (saat) = Σ_t FAA(t) · (1/60)
Eşdeğer Trafo Ömrü (yıl) = normal_ömür_saat / (8760 · ortalama_FAA)
Yaşlanma Maliyeti = (Tüketilen Ömür / normal_ömür_saat) × Trafo_Yenileme_Maliyeti
```

- `normal_ömür_saat = 180.000 saat` (≈ 20.55 yıl, termal-iyileştirilmiş kâğıt).
- `Trafo_Yenileme_Maliyeti` (varsayılan **4.000.000 TL**, 1600 kVA OG trafo + montaj).

**Parasal fayda (Madde 8):** Algoritmasız senaryo sıcak-noktayı 110 °C sınırına
dayar ve eşdeğer ömrü kısaltır. Algoritma sıcak-noktayı güvenli bölgeye çekerek
**erken trafo yenileme/arıza maliyetini erteler**. KPI:
`Önlenen Yaşlanma = Yaşlanma Maliyeti(Algoritmasız) − Yaşlanma Maliyeti(Algoritmalı)`
(100 günlük; yıllık projeksiyon ×365/gün).

> Not: 100 günlük ufukta TL küçük olabilir (trafo dayanıklıdır); asıl gösterge
> **eşdeğer ömrün kat kat uzaması** ve sıcak-noktanın 110 °C altında tutulmasıdır.

---

## 4. Batarya Sağlığı (SOH) → Parasal Karşılık (Madde 8)

Batarya kapasite kaybı, **işlenen enerjiye** ve **C-rate stresine** bağlıdır.

```
C-rate(t)    = P(t) / Kapasite
stres_kWh    = Σ E_dakika · (1 + k · C-rate²)            (k = soh_crate_k = 0.6)
SOH_kaybı(%) = stres_kWh / (Kapasite · çevrim_ömrü) × eol_kayıp × 100
```

- `çevrim_ömrü = 1500` tam çevrim, `eol_kayıp = %20` (ömür sonu kabul edilen kayıp).
- Yüksek C-rate (bodoslama, hızlı şarj) stresi **kuadratik** artırır → daha hızlı yıpranma.

**Parasal fayda (Madde 8):** Korunan SOH, **batarya değişiminin ertelenmesi**dir:

```
Paket Değeri      = Kapasite [kWh] × batarya_maliyeti [TL/kWh]    (varsayılan 4500 TL/kWh)
Korunan Bedel(araç) = (SOH_kaybı_algoritmasız − SOH_kaybı_algoritmalı) × Paket Değeri
Geciktirilen Değişim Bedeli = Σ_araç  Korunan Bedel(araç)
```

---

## 5. Power-Shaving (Rezerv Yük) Yatırım Getirisi

```
Tıraşlama (kW)   = Tepe(Algoritmasız) − Tepe(Algoritmalı)
Tıraşlama (%)    = Tıraşlama / Tepe(Algoritmasız) × 100
İlave İstasyon   = Tıraşlama / ortalama_istasyon_gücü
```

Yorum: Yaratılan headroom ile aynı trafoda **ilave DC istasyon** ve **daha fazla EV**
desteklenebilir; milyonluk trafo yatırımı ertelenir.

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

## Varsayılan Parametre Özeti

| Parametre | Varsayılan | Kaynak/Gerekçe |
|---|---|---|
| Trafo anma | 1600 kVA | Donanım |
| Güç bedeli | 90 TL/kW/ay | EPDK güç bedeli mertebesi |
| Güç aşım ceza katı | 3× | EPDK aşım uygulaması (tarifeye göre değişir) |
| Sözleşme gücü | 1400 kW | Tesis sözleşmesi |
| Batarya maliyeti | 4500 TL/kWh | 2024–25 pak fiyat mertebesi |
| Çevrim ömrü / EOL | 1500 / %20 | Li-ion tipik |
| Trafo yenileme | 4.000.000 TL | 1600 kVA OG trafo + montaj |
| Normal yalıtım ömrü | 180.000 saat | IEEE C57.91 |
| PTF | EPİAŞ MCP (canlı) | EPİAŞ Şeffaflık Platformu |
