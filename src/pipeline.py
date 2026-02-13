"""
Pipeline de datos para dashboard automático (MVP).

Objetivo:
- Recibir Excels "en bruto" de clientes (columnas en español, nombres variables)
- Normalizar columnas a un esquema interno estándar
- Limpiar, calcular KPIs y guardar resultados en /data/processed

Estrategia:
1) Mapeo flexible por sinónimos (ej. "Fecha", "fecha_venta", "día" -> "fecha")
2) Limpieza y tipado robusto
3) Salidas consistentes para el dashboard
"""

from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd


# Esquema interno estándar (siempre igual en el pipeline)
ESQUEMA = ["fecha", "producto", "cantidad", "precio"]


# Sinónimos esperables en Excels reales (puedes ampliar esto con el tiempo)
SINONIMOS = {
    "fecha": [
        "fecha", "dia", "día", "fecha venta", "fecha_venta", "fecha de venta",
        "date", "fecha pedido", "fecha_pedido", "created_at"
    ],
    "producto": [
        "producto", "producto/servicio", "servicio", "articulo", "artículo", "item",
        "nombre producto", "nombre_producto", "descripcion", "descripción", "product"
    ],
    "cantidad": [
        "cantidad", "uds", "unidades", "unidad", "qty", "quantity", "n", "numero", "número"
    ],
    "precio": [
        "precio", "precio unitario", "precio_unitario", "importe", "total", "importe total",
        "valor", "price", "amount"
    ],
}


@dataclass(frozen=True)
class RutasPipeline:
    """Rutas utilizadas por el pipeline."""
    archivo_entrada: Path = Path("data/input/sales.xlsx")
    salida_kpis: Path = Path("data/processed/kpis.parquet")
    salida_limpio: Path = Path("data/processed/ventas_limpias.parquet")


def _normalizar_nombre_columna(nombre: str) -> str:
    """
    Normaliza nombres de columnas para facilitar matching:
    - minúsculas
    - sin espacios repetidos
    - sin caracteres raros
    """
    nombre = str(nombre).strip().lower()
    nombre = nombre.replace("_", " ")
    nombre = re.sub(r"\s+", " ", nombre)
    nombre = nombre.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    nombre = nombre.replace("ñ", "n")
    return nombre


def cargar_excel(ruta: Path) -> pd.DataFrame:
    """Carga un Excel real."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    return pd.read_excel(ruta)


def detectar_mapeo_columnas(columnas: list[str]) -> dict:
    """
    Detecta un mapeo de columnas del Excel del cliente hacia el esquema interno.

    Devuelve dict: {columna_original: columna_estandar}
    """
    cols_norm = {c: _normalizar_nombre_columna(c) for c in columnas}
    mapeo = {}

    for col_original, col_norm in cols_norm.items():
        for estandar, opciones in SINONIMOS.items():
            opciones_norm = [_normalizar_nombre_columna(o) for o in opciones]
            if col_norm in opciones_norm:
                mapeo[col_original] = estandar
                break

    return mapeo


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas del df al esquema estándar: fecha, producto, cantidad, precio.

    Si faltan columnas imprescindibles, lanza error con mensaje claro.
    """
    mapeo = detectar_mapeo_columnas(list(df.columns))
    df = df.rename(columns=mapeo).copy()

    faltan = [c for c in ESQUEMA if c not in df.columns]
    if faltan:
        raise ValueError(
            f"El Excel no tiene columnas suficientes para procesarlo.\n"
            f"Faltan: {faltan}\n"
            f"Columnas detectadas: {list(df.columns)}\n"
            f"Sugerencia: renombra columnas o dime cómo se llaman y lo adaptamos."
        )

    # Nos quedamos con las columnas necesarias
    return df[ESQUEMA].copy()


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica y cálculo de revenue."""
    df = df.dropna(how="all").copy()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["producto"] = df["producto"].astype(str)

    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")

    df = df.dropna(subset=["fecha", "cantidad", "precio"])
    df = df[df["cantidad"] > 0]
    df = df[df["precio"] >= 0]

    df["revenue"] = df["cantidad"] * df["precio"]
    return df


def calcular_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """KPIs básicos del negocio."""
    kpis = {
        "revenue_total": float(df["revenue"].sum()),
        "ordenes_totales": int(len(df)),
        "ticket_medio": float(df["revenue"].mean()) if len(df) else 0.0,
        "unidades_vendidas": float(df["cantidad"].sum()),
        "productos_unicos": int(df["producto"].nunique()),
        "fecha_inicio": df["fecha"].min(),
        "fecha_fin": df["fecha"].max(),
    }
    return pd.DataFrame([kpis])


def guardar_resultados(df_limpio: pd.DataFrame, df_kpis: pd.DataFrame, rutas: RutasPipeline) -> None:
    """Guarda salidas del pipeline."""
    rutas.salida_limpio.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.to_parquet(rutas.salida_limpio, index=False)
    df_kpis.to_parquet(rutas.salida_kpis, index=False)


def ejecutar_pipeline(rutas: RutasPipeline = RutasPipeline()) -> None:
    """Ejecuta pipeline completo."""
    df_raw = cargar_excel(rutas.archivo_entrada)
    df_norm = normalizar_dataframe(df_raw)
    df_limpio = limpiar_datos(df_norm)
    df_kpis = calcular_kpis(df_limpio)
    guardar_resultados(df_limpio, df_kpis, rutas)

    print("✅ Pipeline ejecutado correctamente.")
    print(df_kpis.to_string(index=False))


if __name__ == "__main__":
    ejecutar_pipeline()