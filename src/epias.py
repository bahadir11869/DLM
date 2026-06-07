# -*- coding: utf-8 -*-
"""
epias.py
========
EPIAS (Seffaflik Platformu) API'sinden GERCEK PTF (Piyasa Takas Fiyati / MCP)
verisini ceker.

Kimlik dogrulama akisi (EPIAS yeni API, 2024+):
  1) TGT bileti al:
       POST https://giris.epias.com.tr/cas/v1/tickets
       body: username, password (x-www-form-urlencoded)
       -> donus govdesinde "TGT-..." bileti
  2) PTF/MCP cek:
       POST https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp
       header: "TGT": <bilet>
       body: {"startDate": "...T00:00:00+03:00", "endDate": "...T00:00:00+03:00"}
       -> {"items": [{"date": "...", "price": <TL/MWh>, ...}, ...]}

Kullanim:
  ptf_min, source = get_ptf_minute(days, end_date, username, password, cache_dir)

Tum hatalar (ag yok, gecersiz kimlik, bos yanit) yakalanir; bu durumda
`source="hata: ..."` doner ve cagiran taraf SENTETIK egriye dusebilir.
Boylece dashboard her kosulda calisir.
"""

from __future__ import annotations

import os
import datetime as _dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # requests yoksa
    requests = None

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
MCP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"


def get_tgt(username: str, password: str, timeout: int = 25) -> str:
    """EPIAS CAS sunucusundan TGT bileti alir."""
    if requests is None:
        raise RuntimeError("requests kutuphanesi kurulu degil")
    r = requests.post(
        CAS_URL,
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
        timeout=timeout,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"TGT alinamadi (HTTP {r.status_code}): kimlik bilgilerini kontrol edin")
    # TGT, Location header'inda veya govdede gelebilir.
    loc = r.headers.get("Location", "")
    if "TGT-" in loc:
        return loc.split("/tickets/")[-1].strip()
    body = (r.text or "").strip()
    if body.startswith("TGT-"):
        return body
    # Bazi yanitlar dogrudan bilet stringidir
    if body:
        return body
    raise RuntimeError("TGT yaniti cozumlenemedi")


def fetch_mcp(start_date: str, end_date: str, tgt: str, timeout: int = 60) -> pd.DataFrame:
    """
    [start_date, end_date) araligi icin saatlik MCP/PTF ceker (TL/MWh).
    Tarih formati: 'YYYY-MM-DD'.
    """
    if requests is None:
        raise RuntimeError("requests kutuphanesi kurulu degil")
    body = {
        "startDate": f"{start_date}T00:00:00+03:00",
        "endDate": f"{end_date}T00:00:00+03:00",
    }
    r = requests.post(
        MCP_URL, json=body,
        headers={"TGT": tgt, "Content-Type": "application/json", "Accept": "application/json"},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"MCP cekilemedi (HTTP {r.status_code})")
    data = r.json()
    items = data.get("items", [])
    if not items:
        raise RuntimeError("MCP yaniti bos")
    df = pd.DataFrame(items)
    # 'price' alani PTF/MCP (TL/MWh). 'date' ISO zaman damgasi.
    if "price" not in df.columns:
        raise RuntimeError(f"Beklenen 'price' alani yok: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "price"]]


def get_ptf_minute(days: int, end_date: Optional[str],
                   username: str, password: str,
                   cache_dir: str) -> Tuple[Optional[np.ndarray], str]:
    """
    Son `days` gune ait gercek PTF'yi DAKIKA bazli dizi olarak dondurur (TL/MWh).
    Basarisizlikta (None, "hata: ...") doner.

    Onbellek: ayni (days, end_date) icin data/ altina yazilir.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if not end_date:
        end = _dt.date.today() - _dt.timedelta(days=1)
    else:
        end = _dt.date.fromisoformat(end_date)
    start = end - _dt.timedelta(days=days - 1)
    start_s, end_excl_s = start.isoformat(), (end + _dt.timedelta(days=1)).isoformat()

    cache_path = os.path.join(cache_dir, f"ptf_epias_{start_s}_{end.isoformat()}.parquet")
    # Onbellekten oku
    if os.path.exists(cache_path):
        try:
            dfc = pd.read_parquet(cache_path)
            arr = _to_minute(dfc["price"].values, days)
            return arr, f"onbellek (EPIAS {start_s}..{end.isoformat()})"
        except Exception:
            pass

    if not username or not password:
        return None, "EPIAS kimlik bilgisi girilmedi"
    try:
        tgt = get_tgt(username, password)
        df = fetch_mcp(start_s, end_excl_s, tgt)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
        arr = _to_minute(df["price"].values, days)
        return arr, f"EPIAS canli ({start_s}..{end.isoformat()})"
    except Exception as e:
        return None, f"hata: {e}"


def _to_minute(hourly_price: np.ndarray, days: int) -> np.ndarray:
    """
    Saatlik PTF dizisini (TL/MWh) dakikaya genisletir. Beklenen uzunluk
    days*24; eksik/fazla saatler (DST, eksik gun) kirpilir/doldurulur.
    """
    need = days * 24
    h = np.asarray(hourly_price, dtype=np.float64)
    if h.shape[0] < need:
        # Eksikse son gozlemle ileri-doldur
        pad = np.full(need - h.shape[0], h[-1] if h.shape[0] else 0.0)
        h = np.concatenate([h, pad])
    elif h.shape[0] > need:
        h = h[:need]
    return np.repeat(h, 60)  # her saat 60 dakika -> days*1440
