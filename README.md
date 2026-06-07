# ⚡ Dinamik Yük Dağılımı, Enerji & Maliyet Optimizasyonu Simülasyonu (B2B)

Ticari işletmeler (**AVM** ve **Fabrika/Lojistik Filoları**) için **1600 kVA** ana
trafolu, karışık DC şarj istasyonlarında gerçek zamanlı, dakika bazlı, çok amaçlı
(multi-objective) yük dağılımı, enerji ve maliyet optimizasyonu.

Arka uç **Python (NumPy/Pandas vektörizasyon)**, ön yüz **Streamlit** + **Matplotlib**.

---

## 📁 Proje Yapısı

```
.
├── app.py                  # Streamlit dashboard (Mikro/Makro sekmeleri)
├── baslat.bat              # Windows tek-tik baslatici
├── requirements.txt
├── README.md
├── data/                   # Uretilen zaman serileri (Parquet / CSV)
│   ├── fleet_<senaryo>.parquet      # kalici arac filosu (sabit ID'ler)
│   ├── sessions_<senaryo>.parquet   # sarj oturumlari
│   ├── base_load_<senaryo>.parquet  # tesis baz yuk profili
│   └── market_<senaryo>.parquet     # PTF / SMF egrileri
└── src/
    ├── __init__.py
    ├── config.py           # Tum yapilandirma (donanim, fiyat, termal, SOH)
    ├── data_generator.py   # Arac DB, kalici filo, baz yuk, PTF/SMF
    ├── optimizer.py        # Multi-objective + ±60 kW ramp + trafo korumasi
    └── financials.py       # PTF entegrasyonu, IEEE C57.91 termal, SOH, ROI
```

## 🚀 Çalıştırma

```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```
Windows'ta `baslat.bat` dosyasına çift tıklamak da yeterlidir.
Tarayıcıda `http://localhost:8501` açılır.

---

## 🔑 Temel Modeller

### Donanım & Veri
- **1600 kVA** ana trafo; tesis **baz yükü tepe noktada ~%70 (≈1120 kW)**'e ulaşır,
  kalan kapasite şarj içindir.
- **8 adet karışık DC soket**: 2×200 + 3×180 + 3×120 kW (yapılandırılabilir).
- **23 gerçek araç** (Togg, Tesla, Taycan, Ford E-Transit…) gerçek Net kWh / Max DC kW.
- **Kalıcı filo:** sabit ID'li araçlar 100 gün boyunca tekrar tekrar şarj olur
  (kümülatif SOH analizi için şart). Her gün **stokastik geliş SoC**.

### Fiyatlandırma (iki ölçek)
- **PTF/SMF** (büyük fabrika): EPİAŞ gerçeklerine yakın saatlik **PTF** ve **SMF**
  eğrileri (TL/MWh), hesaplar için **kWh**'ye çevrilir.
- **3-Zamanlı Tarife** (küçük tesis): Gece / Gündüz / Puant (TL/kWh).

### Optimizasyon (`optimizer.py`)
- **±60 kW ramp limiti** (hard constraint) — matematiği dosyada adım adım açıklı.
- **Trafo koruması:** optimize strateji toplam yükü `rated_kw` üstüne çıkarmaz
  (peak-shaving). **Bodoslama** bunu yok sayar → trafo aşırı yüklenir.
- **C-rate (SOH) tavanı**; aciliyet bunu geçersiz kılabilir (servis korunur).
- Şarj istisnasız **%80 SoC**'de biter.
- **α/β/γ** ağırlıkları: Şarj Süresi / SOH Koruma / Maliyet.

### Trafo Termal Ömür (`financials.py`, IEEE C57.91 benzeri)
```
K(t)     = S(t)/S_rated
ΔθTO,ult = ΔθTO,R·((K²·R+1)/(R+1))^n      (üst-yağ nihai artışı)
ΔθTO(t)  : 1. derece gecikme (yağ termal ataleti, τ=180 dk)
ΔθH(t)   = ΔθH,R·K^(2m)                    (sıcak-nokta yağ üstü artışı)
θH(t)    = θ_ortam + ΔθTO + ΔθH
FAA(t)   = exp(15000/383 − 15000/(θH+273)) (110°C'de FAA=1)
LoL      = Σ FAA·Δt  →  %ömür, eşdeğer ömür (yıl), önlenen yaşlanma maliyeti
```

---

## 📊 Dashboard

**BÖLÜM A — 1 Günlük Mikro Analiz**
- Kombine yük + maliyet grafiği: baz yük (fill), Algoritma Öncesi/Sonrası trafo
  yükü, trafo anma çizgisi + **twinx** ile PTF/Tarife eğrisi.
- 1 günlük tasarruf kartları (TL ve %).
- Power-shaving ROI metni (tıraşlama %, yaratılan headroom, +istasyon, +%EV).

**BÖLÜM B — 100 Günlük Makro Analiz**
- Trafo termal ömür tüketimi (kümülatif yaşlanma, Algoritmalı vs Algoritmasız) +
  sıcak-nokta düşüşü, **eşdeğer trafo ömrü (yıl)**, önlenen yaşlanma maliyeti.
- Kümülatif SOH grafiği + geciktirilen batarya değişim bedeli.
- Araç bazlı tablo: SOH düşüşü (algoritmalı vs algoritmasız), toplam şarj süresi,
  maksimum şarj gecikme %'si, korunan batarya bedeli.

---

## 🧭 Mühendislik Notları (dürüstlük)
- En büyük finansal kalemler **demand charge (puant güç bedeli)** ve **SOH
  koruması**dır; bunlar trafonun aşırı yüklenmesini önlemenin doğrudan parasal
  karşılığıdır.
- **Termal ömür** kalemi 100 günlük ufukta TL olarak küçüktür — çünkü 1600 kVA
  trafo bu yüklerde yalıtım ömrünü yavaş tüketir. Asıl termal kazanç **niteldir:**
  bodoslama sıcak-noktayı 110°C yalıtım sınırına dayar; optimize strateji bunu
  güvenli bölgeye çeker ve **eşdeğer trafo ömrünü kat kat uzatır** (ör. 700→2477 yıl).
- **FABRIKA** senaryosunda filo vardiya sonunda eşzamanlı döner; bodoslama akşam
  baz yükü hâlâ yüksekken trafoyu saatlerce aşırı yükler — termal farkın görüldüğü
  yer burasıdır. **AVM**'de dar kalış süresi nedeniyle enerji kalemi ~nötr olabilir;
  değer SOH + demand + power-shaving'dedir.
