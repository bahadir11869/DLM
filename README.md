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
    ├── config.py           # Tum yapilandirma (donanim, fiyat, IEC termal, SOH, diversity)
    ├── data_generator.py   # Arac DB, kalici filo, baz yuk, PTF/SMF, Ankara sicakligi
    ├── optimizer.py        # Multi-objective + ramp + trafo korumasi + cesitlilik
    └── financials.py       # PTF entegrasyonu, IEC 60076-7 termal, 30y omur, SOH, ROI
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
- **1600 kVA** ana trafo; ortalama **güç faktörü cosφ=0.95** → etkin anma **1520 kW**
  (madde 6). Tesis **baz yükü tepe noktada %60 (≈912 kW)**'e ulaşır (madde 9).
- **Karışık DC soket**: 2×200 + 1×180 + 1×120 = **700 kW kurulu** (yapılandırılabilir).
  **Çeşitlilik faktörü 0.60** ile eşzamanlı talep ≈ 420 kW = trafo anmasının **%27.6**'sı
  (hedef %20–30, madde 9) → baz (%60) + istasyon (%27.6) < %100 → **overload yok**.
- **23 gerçek araç** (Togg, Tesla, Taycan, Ford E-Transit…) gerçek Net kWh / Max DC kW.
- **Kalıcı filo:** sabit ID'li araçlar 100 gün boyunca tekrar tekrar şarj olur
  (kümülatif SOH analizi için şart). Her gün **stokastik geliş SoC**.
- **Gerçek Ankara sıcaklıkları** (madde 1): 100 gün **Mayıs başından** başlar
  (en kötü senaryo → ~9 Ağustos, en sıcak bant), termal modele girdi.

### Fiyatlandırma (iki ölçek)
- **PTF/SMF** (büyük fabrika): EPİAŞ gerçeklerine yakın saatlik **PTF** ve **SMF**
  eğrileri (TL/MWh), hesaplar için **kWh**'ye çevrilir.
- **3-Zamanlı Tarife** (küçük tesis): Gece / Gündüz / Puant (TL/kWh).

### Optimizasyon (`optimizer.py`)
- **Ramp limiti** (madde 4): kurulu istasyon gücünün **~%10/dk**'sı (mutlak sabit
  değil; güç-kalitesi/EMS yumuşatması, IEC 61000-3-3/-11). Cihaz kendisi ISO 15118 /
  IEC 61851 ile saniyeler içinde rampa yapar.
- **Çeşitlilik faktörü (madde 6):** algoritma-öncesi (LMS yok) eşzamanlı talep
  = çeşitlilik × kurulu güç (IEC 60364-7-722). Algoritma = aktif yük yönetimi (LMS).
- **Trafo/sözleşme koruması:** optimize strateji toplam yükü trafo/sözleşme gücü
  üstüne çıkarmaz (peak-shaving). **Doğru boyutlandırma** ile naive bile aşmaz (madde 9).
- **C-rate (SOH) tavanı**; aciliyet bunu geçersiz kılabilir (servis korunur).
- Şarj istisnasız **%80 SoC**'de biter. **Şarj verimi %93** (madde 7).
- **α/β/γ** ağırlıkları: Şarj Süresi / SOH Koruma / Maliyet.

### Trafo Termal Ömür (`financials.py`, IEC 60076-7:2018)
```
K(t)     = (Baz+Şarj)/S_anma
Δθo,ult  = Δθor·((1+R·K²)/(1+R))^x         (üst-yağ nihai artışı)
Δθo[t]   : fark denklemi (yağ ataleti, τo=210 dk)
Δθh      = Δθh1 − Δθh2                      (iki zaman sabitli sıcak-nokta gradyanı)
θh(t)    = θa(t) + Δθo + Δθh               (θa = gerçek Ankara sıcaklığı, madde 1)
V(t)     = 2^((θh−98)/6)                    (98°C'de V=1, normal kâğıt)
LoL      = Σ V·Δt  →  30 yıl ekstrapolasyonu (mevsimsel düzeltme), ertelenen
           trafo değişim maliyeti (madde 3)
```

---

## 📊 Dashboard

**BÖLÜM A — 1 Günlük Mikro Analiz**
- Kombine yük + maliyet grafiği: baz yük (fill), Algoritma Öncesi/Sonrası trafo
  yükü, trafo anma çizgisi + **twinx** ile PTF/Tarife eğrisi.
- **Şarj eğrisi — Algoritma Öncesi vs Sonrası** kıyas grafiği (madde 8).
- **Araç bazlı şarj süreleri (bu gün):** her oturum için algoritma öncesi/sonrası
  süre ve uzama — günü takvimden seçerek her **t** günü için (madde 8).
- 1 günlük tasarruf kartları (TL ve %).
- Power-shaving ROI metni (tıraşlama %, yaratılan headroom, +istasyon, +%EV).

**BÖLÜM B — 100 Günlük Makro Analiz**
- **Trafo termal ömür (IEC 60076-7, gerçek Ankara sıcaklıkları)**: kümülatif
  yaşlanma, sıcak-nokta, 100 gün ve **30 yılda tüketilen ömür %'si**, **ertelenen
  trafo değişim maliyeti** (madde 3).
- Kümülatif SOH grafiği + geciktirilen batarya değişim bedeli.
- Araç bazlı tablo: SOH düşüşü (algoritmalı vs algoritmasız), toplam şarj süresi,
  maksimum şarj uzaması, korunan batarya bedeli.

---

## 🧭 Mühendislik Notları (dürüstlük)
- **Madde 9 gereği sistem doğru boyutlandırılmıştır:** baz tepe %60 + çeşitlilikli
  istasyon %20–30 < %100 → **algoritma öncesinde bile trafo aşımı (overload) yoktur.**
  Bu nedenle DLM'in değeri "hard overload önleme" değil; **enerji maliyeti
  (PTF kaydırma)**, **demand charge / peak-shaving**, **termal ömür** ve **SOH**'tur.
- **Termal ömür** kalemi mütevazıdır — çünkü doğru boyutlandırılmış (overload'suz)
  yükte sıcak-nokta IEC referansının (98°C) çoğunlukla altındadır ve trafo ömrü
  **fiziksel tasarım ömrüyle (30 yıl)** sınırlıdır. Model yine de IEC 60076-7 ile
  gerçek Ankara sıcaklıkları altında **30 yıllık ertelenen değişim maliyetini**
  dürüstçe raporlar (madde 3).
- **FABRIKA** senaryosunda filo vardiya sonunda eşzamanlı döner; çeşitlilik faktörü
  bu eşzamanlı talebi gerçekçi biçimde sınırlar (IEC 60364-7-722). **AVM**'de dar
  kalış süresi nedeniyle enerji kalemi ~nötr olabilir; değer SOH + demand +
  power-shaving'dedir.
