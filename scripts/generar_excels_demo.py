from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import random
import uuid

import pandas as pd


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Elevar la calidad de los demos.
# - Crear casos más variados por vertical.
# - Forzar escenarios que tensionen de verdad el motor:
#     * ticket medio,
#     * dependencia de top item,
#     * recurrencia,
#     * reactivación,
#     * día flojo,
#     * calidad de lectura.
# - Mantener datasets suficientemente realistas para demo comercial.
#
# CRÍTICA A LA VERSIÓN ORIGINAL
# - Era buena como generador básico, pero demasiado "limpia" y homogénea.
# - Faltaban tickets reales para probar exactitud vs aproximación.
# - Faltaban escenarios dirigidos: top item dominante, rebooking, clientes dormidos,
#   upsell visible, bundles, etc.
# - No había casi tensión para profile/ranking/history/data_quality.
# =============================================================================


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
    values = [x[0] for x in options]
    weights = [x[1] for x in options]
    return random.choices(values, weights=weights, k=1)[0]



def random_price(base_price: float, volatility: float = 0.10) -> float:
    factor = random.uniform(1 - volatility, 1 + volatility)
    return round(base_price * factor, 2)



def poisson_like(mean_value: float) -> int:
    low = max(1, int(mean_value * 0.70))
    high = max(low + 1, int(mean_value * 1.30))
    return random.randint(low, high)



def maybe(prob: float) -> bool:
    return random.random() < float(prob)



def make_ticket_id(prefix: str, current_date: datetime) -> str:
    rand = uuid.uuid4().hex[:8].upper()
    return f"{prefix}-{current_date.strftime('%Y%m%d')}-{rand}"


# ============================================================
# Estacionalidad / patrones
# ============================================================
def month_seasonality_factor(month: int, business_kind: str) -> float:
    if business_kind == "peluqueria":
        mapping = {
            1: 0.95, 2: 0.92, 3: 1.00, 4: 1.05, 5: 1.08, 6: 1.12,
            7: 1.10, 8: 0.90, 9: 1.00, 10: 1.03, 11: 1.05, 12: 1.15,
        }
        return mapping.get(month, 1.0)

    if business_kind == "cafeteria":
        mapping = {
            1: 1.04, 2: 1.02, 3: 0.99, 4: 1.00, 5: 1.01, 6: 0.98,
            7: 0.95, 8: 0.90, 9: 0.99, 10: 1.02, 11: 1.04, 12: 1.10,
        }
        return mapping.get(month, 1.0)

    if business_kind == "retail":
        mapping = {
            1: 0.94, 2: 0.93, 3: 0.97, 4: 1.00, 5: 1.02, 6: 1.01,
            7: 0.96, 8: 0.90, 9: 1.06, 10: 1.08, 11: 1.12, 12: 1.20,
        }
        return mapping.get(month, 1.0)

    return 1.0



def weekday_factor(weekday: int, business_kind: str) -> float:
    if business_kind == "peluqueria":
        mapping = {0: 0.95, 1: 1.00, 2: 0.68, 3: 1.16, 4: 1.00, 5: 0.80, 6: 1.02}
        return mapping.get(weekday, 1.0)

    if business_kind == "cafeteria":
        mapping = {0: 0.92, 1: 0.95, 2: 0.96, 3: 1.00, 4: 1.06, 5: 1.18, 6: 1.14}
        return mapping.get(weekday, 1.0)

    if business_kind == "retail":
        mapping = {0: 0.94, 1: 0.98, 2: 0.83, 3: 1.02, 4: 0.98, 5: 1.15, 6: 1.06}
        return mapping.get(weekday, 1.0)

    return 1.0



def trend_factor(day_index: int, total_days: int, slope_pct: float) -> float:
    if total_days <= 1:
        return 1.0
    progress = day_index / (total_days - 1)
    return 1.0 + slope_pct * progress



def maybe_campaign_factor(date: datetime, business_kind: str) -> float:
    m = date.month
    d = date.day

    if business_kind == "peluqueria" and m == 5 and 10 <= d <= 25:
        return 1.08

    if business_kind in ("peluqueria", "retail") and m == 8:
        return 0.92

    if m == 12 and d >= 10:
        return 1.12

    return 1.0


# ============================================================
# Helpers de construcción realista
# ============================================================
def build_daily_customers(mean_sales: float, customer_factor_low: float = 0.65, customer_factor_high: float = 1.05) -> int:
    low = max(1, int(mean_sales * customer_factor_low))
    high = max(low + 1, int(mean_sales * customer_factor_high))
    return random.randint(low, high)



def maybe_add_noise_rows(rows: list[dict], *, vertical: str, probability: float = 0.015) -> None:
    """
    Añade unas pocas filas imperfectas para tensionar pipeline/data_quality,
    sin romper el dataset demo principal.
    """
    if not rows:
        return

    noisy_rows: list[dict] = []
    for row in rows:
        if not maybe(probability):
            continue
        x = dict(row)

        r = random.random()
        if r < 0.30:
            # producto vacío / placeholder útil para quality
            first_key = [k for k in x.keys() if k.lower() not in {"fecha", "día", "dia", "date", "ticket", "ticket id", "pedido", "order id", "id ticket"}]
            if first_key:
                prod_key = first_key[0]
                x[prod_key] = ""
        elif r < 0.55:
            # cantidad rara
            for k in x.keys():
                if any(z in k.lower() for z in ["cantidad", "uds", "unidades"]):
                    x[k] = "1,0" if vertical != "retail" else "2"
        elif r < 0.80:
            # precio con símbolo moneda / formato ES
            for k in x.keys():
                if any(z in k.lower() for z in ["precio", "importe", "pvp", "unitario", "total"]):
                    try:
                        val = float(x[k])
                        x[k] = f"€ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    except Exception:
                        pass
        else:
            # ticket vacío parcial
            for k in x.keys():
                if "ticket" in k.lower() or "pedido" in k.lower() or "order" in k.lower():
                    x[k] = ""

        noisy_rows.append(x)

    rows.extend(noisy_rows)


# ============================================================
# Generadores por vertical
# ============================================================
def generate_peluqueria_dataset(
    output_path: Path,
    start_date: datetime,
    months: int = 12,
) -> pd.DataFrame:
    """
    Caso pensado para tensionar:
    - día flojo claro (miércoles)
    - top service dominante
    - ticket medio mejorable
    - bundles / upgrade
    - recurrencia / continuidad
    - reactivación
    - mezcla de ticket exacto (con ticket_id)
    """
    services = [
        ("Tratamiento capilar", 39.0, 1.85),   # top dominante
        ("Corte caballero", 24.0, 1.35),
        ("Corte señora", 31.0, 1.10),
        ("Tinte", 44.0, 0.95),
        ("Peinado", 28.0, 0.75),
        ("Lavado", 11.0, 0.85),
        ("Mascarilla premium", 14.0, 0.36),
        ("Pack corte + tratamiento", 57.0, 0.18),
        ("Upgrade acabado premium", 9.5, 0.22),
    ]

    total_days = months * 30
    rows: list[dict] = []

    # Base pequeña de clientes para simular recurrencia/reactivación
    clients = [f"CL-{i:04d}" for i in range(1, 401)]
    dormant_clients = set(random.sample(clients, 55))

    for i in range(total_days):
        current_date = start_date + timedelta(days=i)

        base_mean = 13.0
        mean_sales = (
            base_mean
            * month_seasonality_factor(current_date.month, "peluqueria")
            * weekday_factor(current_date.weekday(), "peluqueria")
            * trend_factor(i, total_days, slope_pct=0.06)
            * maybe_campaign_factor(current_date, "peluqueria")
        )

        customers_today = build_daily_customers(mean_sales, 0.70, 1.05)

        # Empuje adicional a miércoles flojo con algo de reactivación puntual.
        if current_date.weekday() == 2 and maybe(0.18):
            customers_today += random.randint(2, 5)

        for _ in range(customers_today):
            ticket_id = make_ticket_id("PEL", current_date)
            client_id = random.choice(clients)

            # Reactivación ligera: algunos clientes dormidos vuelven en días flojos.
            reactivated = client_id in dormant_clients and current_date.weekday() == 2 and maybe(0.08)

            main_service = weighted_choice([(s[0], s[2]) for s in services[:6]])
            base_price = next(s[1] for s in services if s[0] == main_service)
            price = random_price(base_price, volatility=0.10)

            # Línea principal
            rows.append(
                {
                    "Fecha de venta": current_date.date(),
                    "Servicio": main_service,
                    "Unidades": 1,
                    "Precio unitario": price,
                    "Ticket ID": ticket_id,
                    "Cliente": client_id,
                    "Canal": "mostrador",
                }
            )

            # Ticket medio mejorable: extras lógicos y upgrades no siempre aceptados.
            if main_service in {"Corte caballero", "Corte señora", "Tinte", "Tratamiento capilar"} and maybe(0.24):
                addon = random.choice(["Mascarilla premium", "Upgrade acabado premium"])
                addon_price = next(s[1] for s in services if s[0] == addon)
                rows.append(
                    {
                        "Fecha de venta": current_date.date(),
                        "Servicio": addon,
                        "Unidades": 1,
                        "Precio unitario": random_price(addon_price, volatility=0.06),
                        "Ticket ID": ticket_id,
                        "Cliente": client_id,
                        "Canal": "mostrador",
                    }
                )

            # Bundle de continuidad real, no masivo.
            if main_service in {"Corte señora", "Tinte", "Tratamiento capilar"} and maybe(0.10):
                bundle_price = next(s[1] for s in services if s[0] == "Pack corte + tratamiento")
                rows.append(
                    {
                        "Fecha de venta": current_date.date(),
                        "Servicio": "Pack corte + tratamiento",
                        "Unidades": 1,
                        "Precio unitario": random_price(bundle_price, volatility=0.05),
                        "Ticket ID": ticket_id,
                        "Cliente": client_id,
                        "Canal": "mostrador",
                    }
                )

            # Rebooking / continuidad: algunos tickets llevan una pseudo-señal de cita siguiente.
            if maybe(0.32) or reactivated:
                rows.append(
                    {
                        "Fecha de venta": current_date.date(),
                        "Servicio": "Próxima visita sugerida",
                        "Unidades": 1,
                        "Precio unitario": 0.0,
                        "Ticket ID": ticket_id,
                        "Cliente": client_id,
                        "Canal": "seguimiento",
                    }
                )

    maybe_add_noise_rows(rows, vertical="peluqueria", probability=0.010)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df



def generate_cafeteria_dataset(
    output_path: Path,
    start_date: datetime,
    months: int = 12,
) -> pd.DataFrame:
    """
    Caso pensado para tensionar:
    - muchas operaciones
    - ticket bajo pero mejorable
    - fin de semana fuerte
    - pairing / complemento guiado
    - segunda visita / recurrencia ligera
    - día flojo entre semana
    """
    products = [
        ("Café", 1.70, 2.35),
        ("Tostada", 2.40, 1.55),
        ("Croissant", 1.95, 1.10),
        ("Refresco", 2.30, 1.00),
        ("Bocadillo", 4.90, 0.95),
        ("Menú del día", 11.90, 0.90),
        ("Zumo natural", 3.50, 0.65),
        ("Postre", 3.20, 0.50),
        ("Combo café + tostada", 3.90, 0.30),
        ("Extra acompañamiento", 1.60, 0.20),
    ]

    total_days = months * 30
    rows: list[dict] = []

    for i in range(total_days):
        current_date = start_date + timedelta(days=i)

        base_mean = 38
        mean_sales = (
            base_mean
            * month_seasonality_factor(current_date.month, "cafeteria")
            * weekday_factor(current_date.weekday(), "cafeteria")
            * trend_factor(i, total_days, slope_pct=0.03)
            * maybe_campaign_factor(current_date, "cafeteria")
        )

        tickets_today = build_daily_customers(mean_sales, 0.70, 1.10)

        for _ in range(tickets_today):
            ticket_id = make_ticket_id("CAF", current_date)

            main = weighted_choice([(p[0], p[2]) for p in products[:8]])
            base_price = next(p[1] for p in products if p[0] == main)

            rows.append(
                {
                    "Fecha": current_date.date(),
                    "Producto": main,
                    "Cantidad": 1,
                    "PVP": random_price(base_price, volatility=0.08),
                    "Pedido": ticket_id,
                    "Canal": random.choice(["barra", "mesa", "terraza"]),
                }
            )

            # Complemento guiado: café->tostada/croissant, menú->postre/refresco, etc.
            if main == "Café" and maybe(0.34):
                addon = random.choice(["Tostada", "Croissant"])
            elif main == "Menú del día" and maybe(0.26):
                addon = random.choice(["Postre", "Refresco"])
            elif main == "Bocadillo" and maybe(0.22):
                addon = random.choice(["Refresco", "Extra acompañamiento"])
            else:
                addon = None

            if addon is not None:
                addon_price = next(p[1] for p in products if p[0] == addon)
                rows.append(
                    {
                        "Fecha": current_date.date(),
                        "Producto": addon,
                        "Cantidad": 1,
                        "PVP": random_price(addon_price, volatility=0.06),
                        "Pedido": ticket_id,
                        "Canal": random.choice(["barra", "mesa", "terraza"]),
                    }
                )

            # Repetición ligera / segunda visita sugerida.
            if current_date.weekday() in {0, 1, 2} and maybe(0.07):
                rows.append(
                    {
                        "Fecha": current_date.date(),
                        "Producto": "Invitación próxima visita",
                        "Cantidad": 1,
                        "PVP": 0.0,
                        "Pedido": ticket_id,
                        "Canal": "seguimiento",
                    }
                )

    maybe_add_noise_rows(rows, vertical="cafeteria", probability=0.012)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df



def generate_retail_dataset(
    output_path: Path,
    start_date: datetime,
    months: int = 12,
) -> pd.DataFrame:
    """
    Caso pensado para tensionar:
    - top item muy dominante
    - venta cruzada
    - bundles de proyecto/solución
    - referencias secundarias
    - ticket con dispersión
    - día flojo claro
    - mezcla de granularidad exacta y parcial
    """
    products = [
        ("Pienso perro", 29.0, 2.20),   # top muy dominante
        ("Snacks", 5.90, 1.05),
        ("Correa", 12.5, 0.68),
        ("Juguete", 8.5, 0.58),
        ("Champú", 9.9, 0.38),
        ("Cama pequeña", 34.0, 0.24),
        ("Arena gato", 11.5, 0.62),
        ("Comedero", 7.2, 0.34),
        ("Pack higiene", 18.0, 0.18),
        ("Kit paseo", 19.5, 0.16),
    ]

    total_days = months * 30
    rows: list[dict] = []

    for i in range(total_days):
        current_date = start_date + timedelta(days=i)

        base_mean = 16
        mean_sales = (
            base_mean
            * month_seasonality_factor(current_date.month, "retail")
            * weekday_factor(current_date.weekday(), "retail")
            * trend_factor(i, total_days, slope_pct=0.05)
            * maybe_campaign_factor(current_date, "retail")
        )

        tickets_today = build_daily_customers(mean_sales, 0.68, 1.00)

        for _ in range(tickets_today):
            ticket_id = make_ticket_id("RET", current_date)
            main = weighted_choice([(p[0], p[2]) for p in products[:8]])
            base_price = next(p[1] for p in products if p[0] == main)

            qty = 1
            if main in {"Snacks", "Arena gato", "Pienso perro"} and maybe(0.22):
                qty = random.choice([2, 2, 3])

            rows.append(
                {
                    "Día": current_date.date(),
                    "Artículo": main,
                    "Uds": qty,
                    "Importe unitario": random_price(base_price, volatility=0.12),
                    "Order ID": ticket_id if maybe(0.78) else "",  # cobertura parcial a propósito
                    "Canal": random.choice(["tienda", "mostrador", "reposición"]),
                }
            )

            # Cross sell útil desde top item dominante
            if main == "Pienso perro" and maybe(0.28):
                addon = random.choice(["Snacks", "Comedero", "Champú"])
                addon_price = next(p[1] for p in products if p[0] == addon)
                rows.append(
                    {
                        "Día": current_date.date(),
                        "Artículo": addon,
                        "Uds": 1,
                        "Importe unitario": random_price(addon_price, volatility=0.08),
                        "Order ID": ticket_id if maybe(0.78) else "",
                        "Canal": "mostrador",
                    }
                )

            # Bundle/solución completa puntual
            if main in {"Correa", "Juguete", "Comedero"} and maybe(0.12):
                addon = random.choice(["Kit paseo", "Pack higiene"])
                addon_price = next(p[1] for p in products if p[0] == addon)
                rows.append(
                    {
                        "Día": current_date.date(),
                        "Artículo": addon,
                        "Uds": 1,
                        "Importe unitario": random_price(addon_price, volatility=0.07),
                        "Order ID": ticket_id if maybe(0.78) else "",
                        "Canal": "mostrador",
                    }
                )

    maybe_add_noise_rows(rows, vertical="retail", probability=0.015)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df


# ============================================================
# Escenarios dirigidos adicionales
# ============================================================
def generate_stress_case_file(output_path: Path) -> pd.DataFrame:
    """
    Dataset pequeño y algo sucio para tensionar pipeline + data_quality.
    No es para demo comercial limpia; es para probar robustez del motor.
    """
    rows = [
        {"Fecha venta": "01/01/2025", "Producto/servicio": "Tratamiento capilar", "Cantidad vendida": "1", "Importe total": "39,00", "Ticket": "A-001"},
        {"Fecha venta": "02/01/2025", "Producto/servicio": "", "Cantidad vendida": "1", "Importe total": "€ 24,00", "Ticket": "A-002"},
        {"Fecha venta": "03/01/2025", "Producto/servicio": "Pienso perro", "Cantidad vendida": "2", "Importe total": "58,00", "Ticket": "A-003"},
        {"Fecha venta": "03/01/2025", "Producto/servicio": "Pienso perro", "Cantidad vendida": "2", "Importe total": "58,00", "Ticket": "A-003"},
        {"Fecha venta": "04/01/2025", "Producto/servicio": "Café", "Cantidad vendida": "1", "Importe total": "0", "Ticket": "A-004"},
        {"Fecha venta": "texto_raro", "Producto/servicio": "Refresco", "Cantidad vendida": "1", "Importe total": "2,20", "Ticket": "A-005"},
        {"Fecha venta": "05/01/2025", "Producto/servicio": "Arena gato", "Cantidad vendida": "-1", "Importe total": "11,50", "Ticket": "A-006"},
        {"Fecha venta": "06/01/2025", "Producto/servicio": "Menú del día", "Cantidad vendida": "1", "Importe total": "(11,90)", "Ticket": ""},
    ]
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
    stress_path = OUTPUT_DIR / "stress_case_demo.xlsx"

    df_pel = generate_peluqueria_dataset(pel_path, start_date=start_date, months=12)
    df_caf = generate_cafeteria_dataset(caf_path, start_date=start_date, months=12)
    df_ret = generate_retail_dataset(ret_path, start_date=start_date, months=12)
    df_stress = generate_stress_case_file(stress_path)

    summarize_dataset(df_pel, "Peluquería / estética")
    summarize_dataset(df_caf, "Cafetería / restauración")
    summarize_dataset(df_ret, "Retail / tienda")
    summarize_dataset(df_stress, "Stress case / calidad")

    print("\nDatasets creados en data/input/:")
    print(f"- {pel_path.name}")
    print(f"- {caf_path.name}")
    print(f"- {ret_path.name}")
    print(f"- {stress_path.name}")
    print("\nSeed usada:", SEED)


if __name__ == "__main__":
    main()