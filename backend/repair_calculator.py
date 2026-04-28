"""
RepairCalculator — bill of materials and labor estimate for facade repair.

Material prices are loaded at runtime from prices_cache.json (written by
price_updater.py during deploy).  If the cache is absent, hardcoded fallbacks
defined in price_updater.PRICE_QUERIES are used.

Consumption rates (kg or l per m²) are defined here and never change;
only prices vary by market.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Material catalog  —  id → (display_name_ru, unit)
# Prices come from price_updater.load_prices(); see fallbacks in PRICE_QUERIES.
# ─────────────────────────────────────────────────────────────────────────────

MATERIAL_CATALOG: Dict[str, Tuple[str, str]] = {
    "facade_putty":         ("Шпатлёвка фасадная",                   "кг"),
    "primer_deep":          ("Грунтовка глубокого проникновения",     "л"),
    "paint_facade":         ("Краска фасадная",                       "л"),
    "repair_compound":      ("Ремонтный состав (цементный)",           "кг"),
    "reinforcing_mesh":     ("Армирующая сетка фасадная",             "м²"),
    "cement_plaster":       ("Штукатурка цементная фасадная",         "кг"),
    "waterproof_compound":  ("Гидроизоляционный состав",              "кг"),
    "anti_salt":            ("Антисоль (очиститель высолов)",         "л"),
    "hydrophobizer":        ("Гидрофобизатор фасадный",               "л"),
    "antiseptic":           ("Антисептик фасадный (биозащита)",       "л"),
    "rust_converter":       ("Преобразователь ржавчины",              "л"),
    "anticorrosion_primer": ("Грунт антикоррозийный",                 "л"),
    "metal_paint":          ("Краска по металлу",                     "л"),
    "wood_antiseptic":      ("Антисептик для дерева",                 "л"),
    "wood_putty":           ("Шпатлёвка по дереву",                   "кг"),
    "wood_paint":           ("Краска/лак по дереву",                  "л"),
    "glass_unit":           ("Стеклопакет двухкамерный",              "шт"),
    "sealant":              ("Герметик силиконовый",                  "шт"),
    "welding_materials":    ("Сварочные материалы",                   "комп"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Consumption rates  —  defect_type → [(material_id, rate_per_m2), …]
# Rates are in the unit declared in MATERIAL_CATALOG (кг/м², л/м², шт/м², …)
# ─────────────────────────────────────────────────────────────────────────────

MATERIAL_RATES: Dict[str, List[Tuple[str, float]]] = {
    "crack_surface": [
        ("facade_putty",  0.30),
        ("primer_deep",   0.15),
        ("paint_facade",  0.25),
    ],
    "crack_deep": [
        ("repair_compound",  1.5),
        ("reinforcing_mesh", 1.1),
        ("primer_deep",      0.15),
        ("facade_putty",     0.5),
        ("paint_facade",     0.25),
    ],
    "peeling": [
        ("cement_plaster",   14.0),
        ("reinforcing_mesh",  1.1),
        ("primer_deep",       0.20),
        ("facade_putty",      0.80),
        ("paint_facade",      0.25),
    ],
    "exposed_brick": [
        ("cement_plaster",   14.0),
        ("reinforcing_mesh",  1.1),
        ("primer_deep",       0.20),
        ("facade_putty",      0.80),
        ("paint_facade",      0.25),
    ],
    "water_damage": [
        ("waterproof_compound", 1.00),
        ("primer_deep",         0.20),
        ("cement_plaster",      2.50),   # surface repair, not full coat
        ("paint_facade",        0.25),
    ],
    "efflorescence": [
        ("anti_salt",     0.30),
        ("hydrophobizer", 0.25),
    ],
    "moss": [
        ("antiseptic",   0.30),
        ("primer_deep",  0.15),
        ("paint_facade", 0.25),
    ],
    "rust": [
        ("rust_converter",       0.15),
        ("anticorrosion_primer", 0.12),
        ("paint_facade",         0.25),
    ],
    "spalling": [
        ("repair_compound",  2.0),
        ("reinforcing_mesh", 1.1),
        ("primer_deep",      0.20),
        ("facade_putty",     0.80),
        ("paint_facade",     0.25),
    ],
    "broken_glass": [
        ("glass_unit", 0.67),   # ~1 unit per 1.5 m²
        ("sealant",    0.67),
    ],
    "damaged_wood": [
        ("wood_antiseptic", 0.15),
        ("wood_putty",      0.50),
        ("wood_paint",      0.15),
    ],
    "rusty_metal": [
        ("rust_converter",       0.20),
        ("anticorrosion_primer", 0.15),
        ("metal_paint",          0.12),
    ],
    "damaged_railing": [
        ("rust_converter",       0.20),
        ("anticorrosion_primer", 0.15),
        ("metal_paint",          0.12),
        ("welding_materials",    0.10),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Labor rates  —  defect_type → (display_ru, unit, rub_per_unit, hours_per_unit)
# ─────────────────────────────────────────────────────────────────────────────

LABOR_RATES: Dict[str, Tuple[str, str, float, float]] = {
    "crack_surface":  ("Заделка поверхностных трещин",            "м.п.",  450,  0.5),
    "crack_deep":     ("Ремонт глубоких трещин с армированием",   "м.п.", 1200,  1.5),
    "peeling":        ("Восстановление штукатурного слоя",        "м²",  1200,  2.0),
    "exposed_brick":  ("Оштукатуривание оголённой кладки",        "м²",  1500,  2.5),
    "water_damage":   ("Устранение последствий протечек",         "м²",   900,  1.5),
    "efflorescence":  ("Удаление высолов и гидрофобизация",       "м²",   450,  0.8),
    "moss":           ("Биоочистка и антисептирование",           "м²",   380,  0.7),
    "rust":           ("Антикоррозийная обработка",               "м²",   500,  1.0),
    "spalling":       ("Восстановление бетонного покрытия",       "м²",  1800,  2.5),
    "broken_glass":   ("Замена стеклопакетов",                    "шт",  4500,  2.0),
    "damaged_wood":   ("Ремонт деревянных элементов",             "м²",  1000,  1.5),
    "rusty_metal":    ("Антикоррозийная обработка металла",       "м²",   650,  1.2),
    "damaged_railing":("Ремонт ограждений балконов",              "м.п.", 1800,  2.0),
}

SCAFFOLDING_COST_PER_FLOOR = 45_000   # RUB per floor
SCAFFOLDING_SETUP_HOURS = 8
MATERIAL_SURPLUS = 0.10               # 10% запас
COUNTABLE_UNITS = {"шт", "комп"}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepairMaterial:
    name: str
    name_display: str
    unit: str
    price_per_unit: float
    quantity: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.price_per_unit * self.quantity


@dataclass
class RepairWork:
    name: str
    name_display: str
    unit: str
    price_per_unit: float
    quantity: float = 0.0
    norm_hours_per_unit: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.price_per_unit * self.quantity

    @property
    def total_hours(self) -> float:
        return self.norm_hours_per_unit * self.quantity


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def px_to_m2(area_px: int, total_px: int, total_m2: float) -> float:
    if total_px == 0:
        return 0.0
    return (area_px / total_px) * total_m2


def estimate_floors(total_area_m2: float,
                    avg_floor_height_m: float = 3.0,
                    avg_floor_width_m: float = 15.0) -> int:
    if avg_floor_width_m == 0:
        return 1
    facade_height = total_area_m2 / avg_floor_width_m
    return max(1, math.ceil(facade_height / avg_floor_height_m))


def _get_prices() -> Dict[str, float]:
    """Load live prices from cache; gracefully fall back to defaults."""
    try:
        from price_updater import load_prices
        return load_prices()
    except Exception as e:
        logger.warning(f"price_updater unavailable ({e}) — using hardcoded defaults.")
        from price_updater import PRICE_QUERIES
        return {mid: float(fb) for mid, (_, _, fb) in PRICE_QUERIES.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Calculator
# ─────────────────────────────────────────────────────────────────────────────

class RepairCalculator:

    def __init__(self, total_area_m2: float = 100.0):
        self.total_area_m2 = total_area_m2

    def calculate(
        self,
        damages: List[dict],
        total_area_px: int,
        layer_analysis: Dict[str, dict],
    ) -> dict:
        prices = _get_prices()

        materials_agg: Dict[str, RepairMaterial] = {}
        labor_items: List[RepairWork] = []

        for damage in damages:
            dtype = damage["type"]
            area_px = damage["area_px"]
            area_m2 = px_to_m2(area_px, total_area_px, self.total_area_m2)

            if area_m2 < 0.01:
                continue

            # Crack depth changes the repair spec
            effective_type = dtype
            if dtype == "crack":
                crack_depth = (
                    damage.get("crack_depth") or
                    layer_analysis.get("crack", {}).get("crack_depth", "surface")
                )
                effective_type = f"crack_{crack_depth}"

            # ── Materials ────────────────────────────────────────────────────
            for mat_id, rate in MATERIAL_RATES.get(effective_type, []):
                if mat_id not in MATERIAL_CATALOG:
                    continue
                display, unit = MATERIAL_CATALOG[mat_id]
                price = prices.get(mat_id, 500.0)
                quantity = rate * area_m2
                if mat_id in materials_agg:
                    materials_agg[mat_id].quantity += quantity
                else:
                    materials_agg[mat_id] = RepairMaterial(
                        name=mat_id,
                        name_display=display,
                        unit=unit,
                        price_per_unit=price,
                        quantity=quantity,
                    )

            # ── Labor ─────────────────────────────────────────────────────────
            if effective_type in LABOR_RATES:
                l_display, l_unit, l_price, l_hours = LABOR_RATES[effective_type]
                labor_items.append(RepairWork(
                    name=effective_type,
                    name_display=l_display,
                    unit=l_unit,
                    price_per_unit=l_price,
                    quantity=round(area_m2, 1),
                    norm_hours_per_unit=l_hours,
                ))

        # Apply surplus + round countable items up
        for mat in materials_agg.values():
            mat.quantity *= (1 + MATERIAL_SURPLUS)
            if mat.unit in COUNTABLE_UNITS:
                mat.quantity = math.ceil(mat.quantity)
            else:
                mat.quantity = round(mat.quantity, 2)

        # Scaffolding
        floors = estimate_floors(self.total_area_m2)
        scaffolding_cost = floors * SCAFFOLDING_COST_PER_FLOOR
        scaffolding_hours = floors * SCAFFOLDING_SETUP_HOURS

        # Totals
        materials_total = sum(m.total_cost for m in materials_agg.values())
        labor_total = sum(l.total_cost for l in labor_items)
        total_hours = sum(l.total_hours for l in labor_items) + scaffolding_hours
        grand_total = materials_total + labor_total + scaffolding_cost

        return {
            "materials": [
                {
                    "name": m.name,
                    "name_display": m.name_display,
                    "unit": m.unit,
                    "quantity": round(m.quantity, 2),
                    "price_per_unit": m.price_per_unit,
                    "total_cost": round(m.total_cost),
                }
                for m in sorted(materials_agg.values(), key=lambda x: x.total_cost, reverse=True)
            ],
            "labor": [
                {
                    "name": l.name,
                    "name_display": l.name_display,
                    "unit": l.unit,
                    "quantity": l.quantity,
                    "price_per_unit": l.price_per_unit,
                    "total_cost": round(l.total_cost),
                    "norm_hours": round(l.total_hours, 1),
                }
                for l in sorted(labor_items, key=lambda x: x.total_cost, reverse=True)
            ],
            "scaffolding": {
                "floors": floors,
                "cost": scaffolding_cost,
                "setup_hours": scaffolding_hours,
            },
            "summary": {
                "materials_total": round(materials_total),
                "labor_total": round(labor_total),
                "scaffolding_total": scaffolding_cost,
                "grand_total": round(grand_total),
                "total_work_hours": round(total_hours, 1),
                "estimated_days": max(1, math.ceil(total_hours / 8)),
                "currency": "₽",
            },
            "costs_for_flutter": _build_flutter_costs(materials_agg, labor_items, scaffolding_cost),
        }


def _build_flutter_costs(
    materials: Dict[str, RepairMaterial],
    labor: List[RepairWork],
    scaffolding: float,
) -> List[dict]:
    items = []
    mat_total = sum(m.total_cost for m in materials.values())
    if mat_total > 0:
        items.append({
            "category": "Строительные материалы",
            "description": f"{len(materials)} наименований с учётом запаса 10%",
            "cost": round(mat_total),
            "unit": "₽",
        })
    for l in sorted(labor, key=lambda x: x.total_cost, reverse=True):
        if l.total_cost > 0:
            items.append({
                "category": l.name_display,
                "description": f"{l.quantity} {l.unit}",
                "cost": round(l.total_cost),
                "unit": "₽",
            })
    if scaffolding > 0:
        items.append({
            "category": "Леса и оборудование",
            "description": "Монтаж/демонтаж строительных лесов",
            "cost": round(scaffolding),
            "unit": "₽",
        })
    return items
