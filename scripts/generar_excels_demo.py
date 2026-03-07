from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import random
import math

import pandas as pd


# ============================================================
# Config general
# ============================================================
OUTPUT_DIR = Path("data/input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)


# ============================================================
# Helpers generales
# ============================================================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_choice(options: list[tuple[str, float]]) -> str:
    """
    options = [(valor, peso), ...]
    """
    values = [x[0] for x in options]
    weights = [x[1] for x in options]
    return random.choices(values, weights=weights, k=1)[0]


def month_seasonality_factor(month: int, business_kind: str) -> float:
    """
    Estacionalidad simple por vertical.
    """
    if business_kind == "peluqueria":
        # más movimiento en primavera/verano y diciembre
        mapping = {
            1: 0.95,
            2: 0.92,
            3: 1.00,
            4: 1.05,
            5: 1.08,
            6: 1.12,
            7: 1.10,
            8: 0.90,
            9: 1.00,
            10: 1.03,
            11: 1.05,
            12: 1.15,
        }
        return mapping.get(month, 1.0)

    if business_kind == "cafeteria":
        # más estable, con algo más de diciembre y meses fríos
        mapping = {
            1: 1.04,
            2: 1.02,
            3: 0.99,
            4: 1.00,
            5: 1.01,
            6: 0.98,
            7: 0.95,
            8: 0.90,
            9: 0.99,
            10: 1.02,
            11: 1.04,
            12: 1.10,
        }
        return mapping.get(month, 1.0)

    if business_kind == "retail":
        # más fuerte en vuelta al cole / otoño / navidad
        mapping = {
            1: 0.94,
            2: 0.93,
            3: 0.97,
            4: 1.00,
            5: 1.02,
            6: 1.01,
            7: 0.96,
            8: 0.90,
            9: 1.06,
            10: 1.08,
            11: 1.12,
            12: 1.20,
        }
        return mapping.get(month, 1.0)

    return 1.0


def weekday_factor(weekday: int, business_kind: str) -> float:
    """
    weekday: Monday=0 ... Sunday=6
    """
    if business_kind == "peluqueria":
        # Miércoles flojo, jueves fuerte
        mapping = {
            0: 0.95,  # lunes
            1: 1.00,  # martes
            2: 0.72,  # miércoles
            3: 1.14,  # jueves
            4: 1.00,  # viernes
            5: 0.78,  # sábado
            6: 1.04,  # domingo
        }
        return mapping.get(weekday, 1.0)

    if business_kind == "cafeteria":
        # finde más fuerte
        mapping = {
            0: 0.92,
            1: 0.95,
            2: 0.96,
            3: 1.00,
            4: 1.06,
            5: 1.18,
            6: 1.14,
        }
        return mapping.get(weekday, 1.0)

    if business_kind == "retail":
        # mitad de semana algo más floja, sábado más fuerte
        mapping = {
            0: 0.94,
            1: 0.98,
            2: 0.85,
            3: 1.02,
            4: 0.98,
            5: 1.15,
            6: 1.06,
        }
        return mapping.get(weekday, 1.0)

    return 1.0


def trend_factor(day_index: int, total_days: int, slope_pct: float) -> float:
    """
    slope_pct = +0.10 -> termina ~10% arriba
    """
    if total_days <= 1:
        return 1.0
    progress = day_index / (total_days - 1)
    return 1.0 + slope_pct * progress


def maybe_campaign_factor(date: datetime, business_kind: str) -> float:
    """
    Mete pequeñas campañas o baches localizados.
    """
    m = date.month
    d = date.day

    # campaña simple de primavera
    if business_kind == "peluqueria" and m == 5 and 10 <= d <= 25:
        return 1.08

    # bache de agosto
    if business_kind in ("peluqueria", "retail") and m == 8:
        return 0.92

    # navidad
    if m == 12 and d >= 10:
        return 1.12

    return 1.0


def random_price(base_price: float, volatility: float = 0.10) -> float:
    """
    Variación porcentual alrededor del precio base.
    """
    factor = random.uniform(1 - volatility, 1 + volatility)
    return round(base_price * factor, 2)


def poisson_like(mean_value: float) -> int:
    """
    Aproximación simple sin numpy.
    """
    low = max(1, int(mean_value * 0.70))
    high = max(low + 1, int(mean_value * 1.30))
    return random.randint(low, high)


# ============================================================
# Generadores por vertical
# ============================================================
def generate_peluqueria_dataset(
    output_path: Path,
    start_date: datetime,
    months: int = 12,
) -> pd.DataFrame:
    """
    Caso pensado para:
    - día flojo claro
    - top service dominante
    - recurrencia / continuidad
    - ticket medio mejorable
    """
    services = [
        ("Corte caballero", 24.0, 1.35),
        ("Corte señora", 31.0, 1.15),
        ("Tinte", 42.0, 1.00),
        ("Lavado", 11.0, 0.90),
        ("Peinado", 28.0, 0.80),
        ("Tratamiento", 38.0, 1.55),  # top fuerte
        ("Pack corte + tratamiento", 56.0, 0.22),
        ("Mascarilla premium", 14.0, 0.40),
    ]

    total_days = months * 30
    rows: list[dict] = []

    for i in range(total_days):
        current_date = start_date + timedelta(days=i)

        base_mean = 12.5
        mean_sales = (
            base_mean
            * month_seasonality_factor(current_date.month, "peluqueria")
            * weekday_factor(current_date.weekday(), "peluqueria")
            * trend_factor(i, total_days, slope_pct=0.06)
            * maybe_campaign_factor(current_date, "peluqueria")
        )

        sales_count = poisson_like(mean_sales)

        for _ in range(sales_count):
            service_name = weighted_choice([(s[0], s[2]) for s in services])
            base_price = next(s[1] for s in services if s[0] == service_name)

            # En peluquería normalmente cantidad = 1, pero metemos algunos extras
            quantity = 1
            if service_name in ("Mascarilla premium",) and random.random() < 0.15:
                quantity = 2

            price = random_price(base_price, volatility=0.10)

            # Simula algunos tickets con pack en el día flojo
            rows.append(
                {
                    "Fecha de venta": current_date.date(),
                    "Servicio": service_name,
                    "Unidades": quantity,
                    "Precio unitario": price,
                }
            )

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df


def generate_cafeteria_dataset(
    output_path: Path,
    start_date: datetime,
    months: int = 12,
) -> pd.DataFrame:
    """
    Caso pensado para:
    - más operaciones
    - ticket menor
    - fin de semana fuerte
    - recomendación de complemento
    """
    products = [
        ("Café", 1.70, 2.4),
        ("Tostada", 2.40, 1.6),
        ("Croissant", 1.90, 1.2),
        ("Refresco", 2.20, 1.1),
        ("Bocadillo", 4.80, 1.0),
        ("Menú del día", 11.90, 0.9),
        ("Zumo natural", 3.40, 0.7),
        ("Postre", 3.10, 0.55),
        ("Combo café + tostada", 3.80, 0.35),
    ]

    total_days = months * 30
    rows: list[dict] = []

    for i in range(total_days):
        current_date = start_date + timedelta(days=i)

        base_mean = 36
        mean_sales = (
            base_mean
            * month_seasonality_factor(current_date.month, "cafeteria")
            * weekday_factor(current_date.weekday(), "cafeteria")
            * trend_factor(i, total_days, slope_pct=0.03)
            * maybe_campaign_factor(current_date, "cafeteria")
        )

        sales_count = poisson_like(mean_sales)

        for _ in range(sales_count):
            product_name = weighted_choice([(p[0], p[2]) for p in products])
            base_price = next(p[1] for p in products if p[0] == product_name)

            quantity = 1
            if product_name in ("Café", "Refresco") and random.random() < 0.08:
                quantity = 2

            price = random_price(base_price, volatility=0.08)

            rows.append(
                {
                    "Fecha": current_date.date(),
                    "Producto": product_name,
                    "Cantidad": quantity,
                    "PVP": price,
                }
            )

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df


def generate_retail_dataset(
    output_path: Path,
    start_date: datetime,
    months: int = 12,
) -> pd.DataFrame:
    """
    Caso pensado para:
    - dependencia de top item
    - venta cruzada
    - referencias secundarias
    - ticket con más dispersión
    """
    products = [
        ("Pienso perro", 29.0, 1.90),   # top muy dominante
        ("Snacks", 5.90, 1.10),
        ("Correa", 12.5, 0.70),
        ("Juguete", 8.5, 0.60),
        ("Champú", 9.9, 0.42),
        ("Cama pequeña", 34.0, 0.28),
        ("Arena gato", 11.5, 0.65),
        ("Comedero", 7.2, 0.35),
        ("Pack higiene", 18.0, 0.22),
    ]

    total_days = months * 30
    rows: list[dict] = []

    for i in range(total_days):
        current_date = start_date + timedelta(days=i)

        base_mean = 15
        mean_sales = (
            base_mean
            * month_seasonality_factor(current_date.month, "retail")
            * weekday_factor(current_date.weekday(), "retail")
            * trend_factor(i, total_days, slope_pct=0.05)
            * maybe_campaign_factor(current_date, "retail")
        )

        sales_count = poisson_like(mean_sales)

        for _ in range(sales_count):
            product_name = weighted_choice([(p[0], p[2]) for p in products])
            base_price = next(p[1] for p in products if p[0] == product_name)

            # En retail hay algo más de cantidad >1
            quantity = 1
            if product_name in ("Snacks", "Arena gato", "Pienso perro") and random.random() < 0.22:
                quantity = random.choice([2, 2, 3])

            price = random_price(base_price, volatility=0.12)

            rows.append(
                {
                    "Día": current_date.date(),
                    "Artículo": product_name,
                    "Uds": quantity,
                    "Importe unitario": price,
                }
            )

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df


# ============================================================
# Resumen útil
# ============================================================
def summarize_dataset(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"Filas: {len(df):,}".replace(",", "."))
    print("Primeras columnas:", list(df.columns))
    print("Rango fechas:", df.iloc[:, 0].min(), "→", df.iloc[:, 0].max())


# ============================================================
# Main
# ============================================================
def main() -> None:
    start_date = datetime(2025, 1, 1)

    pel_path = OUTPUT_DIR / "peluqueria_demo_12m.xlsx"
    caf_path = OUTPUT_DIR / "cafeteria_demo_12m.xlsx"
    ret_path = OUTPUT_DIR / "retail_mascotas_demo_12m.xlsx"

    df_pel = generate_peluqueria_dataset(pel_path, start_date=start_date, months=12)
    df_caf = generate_cafeteria_dataset(caf_path, start_date=start_date, months=12)
    df_ret = generate_retail_dataset(ret_path, start_date=start_date, months=12)

    summarize_dataset(df_pel, "Peluquería / estética")
    summarize_dataset(df_caf, "Cafetería / restauración")
    summarize_dataset(df_ret, "Retail / tienda")

    print("\nDatasets creados en data/input/:")
    print(f"- {pel_path.name}")
    print(f"- {caf_path.name}")
    print(f"- {ret_path.name}")
    print("\nSeed usada:", SEED)


if __name__ == "__main__":
    main()