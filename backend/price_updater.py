"""
Price updater — fetches current facade repair material prices from leroymerlin.ru.

Usage:
    python price_updater.py           # update if cache > 7 days old
    python price_updater.py --force   # always refresh

Prices are cached in prices_cache.json next to this file.
repair_calculator.py calls load_prices() to get the latest values,
falling back to hardcoded defaults if the cache is missing.
"""

import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "prices_cache.json"
CACHE_TTL_DAYS = 7

# material_id → (search_query_ru, unit, fallback_price_rub)
# Fallback prices: Russian retail (Leroy Merlin / Petrochem / 2024)
PRICE_QUERIES: Dict[str, tuple] = {
    "facade_putty":         ("шпатлёвка фасадная акриловая",             "кг",    450),
    "primer_deep":          ("грунтовка глубокого проникновения",         "л",     200),
    "paint_facade":         ("краска фасадная акриловая",                 "л",     550),
    "repair_compound":      ("ремонтный состав цементный фасадный",       "кг",    380),
    "reinforcing_mesh":     ("сетка армирующая стеклотканевая фасад",     "м²",    85),
    "cement_plaster":       ("штукатурка цементная фасадная",             "кг",    50),
    "waterproof_compound":  ("гидроизоляционный состав обмазочный",       "кг",    650),
    "anti_salt":            ("антисоль очиститель высолов фасад",         "л",     380),
    "hydrophobizer":        ("гидрофобизатор фасадный концентрат",        "л",     420),
    "antiseptic":           ("антисептик фасадный биозащита",             "л",     320),
    "rust_converter":       ("преобразователь ржавчины",                  "л",     380),
    "anticorrosion_primer": ("грунт антикоррозийный алкидный",            "л",     320),
    "metal_paint":          ("краска по металлу",                         "л",     520),
    "wood_antiseptic":      ("антисептик защита дерева",                  "л",     320),
    "wood_putty":           ("шпатлёвка по дереву акриловая",             "кг",    280),
    "wood_paint":           ("краска лак для дерева фасадный",            "л",     650),
    "glass_unit":           ("стеклопакет двухкамерный",                  "шт",   7500),
    "sealant":              ("герметик силиконовый нейтральный прозрачный","шт",    350),
    "welding_materials":    ("сварочные электроды",                       "комп", 4500),
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}


def _extract_prices_from_html(html: str) -> list:
    """Pull numeric price values out of a leroymerlin.ru search page."""
    found = []

    # Primary: __NEXT_DATA__ JSON block embedded in every Next.js page
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if m:
        try:
            raw = m.group(1)
            # All "price": NUMBER occurrences inside the JSON blob
            hits = re.findall(
                r'"(?:price|basePrice|regularPrice|salePrice|currentPrice)"\s*:\s*(\d+(?:[.,]\d+)?)',
                raw,
            )
            for h in hits:
                v = float(h.replace(",", "."))
                if 20 <= v <= 200_000:
                    found.append(v)
        except Exception:
            pass

    # Secondary: bare JSON-like price fields anywhere in the page
    if not found:
        hits = re.findall(r'"price"\s*:\s*(\d+(?:[.,]\d+)?)', html)
        for h in hits:
            v = float(h.replace(",", "."))
            if 20 <= v <= 200_000:
                found.append(v)

    return found


def _fetch_price(query: str, fallback: float) -> float:
    """Search leroymerlin.ru and return the median price across first results."""
    try:
        import requests as req
        encoded = req.utils.quote(query)
        url = f"https://leroymerlin.ru/search/?q={encoded}&sort=popularity"
        r = req.get(url, headers=_HEADERS, timeout=14, allow_redirects=True)
        if r.ok:
            prices = _extract_prices_from_html(r.text)
            if prices:
                prices.sort()
                sample = prices[:8]          # top-8 to reduce outlier weight
                median = statistics.median(sample)
                logger.info(f"  ✓  {query!r:48s} → {median:>7.0f} ₽  ({len(sample)} hits)")
                return round(median)
        logger.warning(f"  ✗  {query!r} → HTTP {r.status_code}, fallback {fallback:.0f} ₽")
    except ImportError:
        logger.error("  requests not installed — pip install requests")
    except Exception as e:
        logger.warning(f"  ✗  {query!r} → {e!s:.80s}, fallback {fallback:.0f} ₽")
    return fallback


def update_prices(force: bool = False) -> Dict[str, float]:
    """
    Fetch prices from leroymerlin.ru and save to prices_cache.json.

    Skips if cache is fresher than CACHE_TTL_DAYS (unless force=True).
    Always returns the current price dict (fetched or cached).
    """
    if not force and CACHE_PATH.exists():
        age_days = (time.time() - CACHE_PATH.stat().st_mtime) / 86400
        if age_days < CACHE_TTL_DAYS:
            logger.info(
                f"Price cache is {age_days:.1f} days old "
                f"(TTL {CACHE_TTL_DAYS} days) — skipping. Use --force to refresh."
            )
            return load_prices()

    logger.info("Fetching material prices from leroymerlin.ru …")
    prices: Dict[str, float] = {}
    for mat_id, (query, unit, fallback) in PRICE_QUERIES.items():
        prices[mat_id] = _fetch_price(query, fallback)
        time.sleep(0.7)   # polite crawl rate

    CACHE_PATH.write_text(
        json.dumps(prices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Prices saved → {CACHE_PATH}")
    return prices


def load_prices() -> Dict[str, float]:
    """
    Load prices from cache.
    Returns hardcoded fallbacks if the cache file is missing or corrupted.
    """
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Price cache unreadable ({e}) — using defaults.")
    return {mid: float(fb) for mid, (_, _, fb) in PRICE_QUERIES.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    update_prices(force="--force" in sys.argv)
