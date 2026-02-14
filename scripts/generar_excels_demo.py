import pandas as pd
from pathlib import Path
import random
from datetime import datetime, timedelta

Path("data/input").mkdir(parents=True, exist_ok=True)

def generar_dataset(nombre_archivo, items, precio_min, precio_max, col_fecha="Fecha de venta", col_item="Servicio"):
    filas = []
    fecha_inicio = datetime(2026, 1, 1)

    for i in range(90):  # 3 meses
        fecha = fecha_inicio + timedelta(days=i)
        ventas_dia = random.randint(5, 20)

        for _ in range(ventas_dia):
            item = random.choice(items)
            cantidad = random.randint(1, 3)
            precio = random.randint(precio_min, precio_max)
            filas.append({
                col_fecha: fecha.date(),
                col_item: item,
                "Unidades": cantidad,
                "Precio unitario": precio
            })

    df = pd.DataFrame(filas)
    df.to_excel(f"data/input/{nombre_archivo}", index=False)

# Peluquería
generar_dataset(
    "peluqueria.xlsx",
    ["Corte caballero", "Corte señora", "Tinte", "Lavado", "Peinado", "Tratamiento"],
    10, 45,
    col_fecha="Fecha de venta",
    col_item="Servicio"
)

# Cafetería
generar_dataset(
    "cafeteria.xlsx",
    ["Café", "Tostada", "Bocadillo", "Refresco", "Menú del día", "Croissant"],
    2, 12,
    col_fecha="Fecha de venta",
    col_item="Producto"
)

# Tienda de mascotas
generar_dataset(
    "mascotas.xlsx",
    ["Pienso perro", "Correa", "Juguete", "Champú", "Snacks", "Cama pequeña"],
    5, 35,
    col_fecha="Fecha de venta",
    col_item="Producto"
)

print("Datasets creados en data/input/")
