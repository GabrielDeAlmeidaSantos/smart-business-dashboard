# app/analytics/history.py
from __future__ import annotations

"""
LEGACY COMPAT LAYER (NO USAR PARA NUEVO CÓDIGO)

Este módulo existía con un HistoryStore antiguo (no atómico, esquema distinto).
Para evitar romper imports legacy, lo convertimos en wrapper del HistoryStore nuevo
(app/analytics/history_store.py), que escribe de forma atómica y con backup.

Si encuentras imports antiguos:
  from .history import HistoryStore
se mantendrán funcionando, pero internamente usan el store nuevo.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal, Dict, List

from .history_store import HistoryStore as _NewHistoryStore

Outcome = Literal["unknown", "improved", "not_improved"]


@dataclass
class HistoryStore:
    """
    Wrapper compatible con la interfaz legacy, respaldado por HistoryStore nuevo.

    IMPORTANTE:
    - El store nuevo usa schema v1:
        {"schema_version":1, "client_id":..., "recommendations":[{period, generated_at, meta, items:[{insight_id,status,outcome,note,updated_at}]}]}
    - Este wrapper traduce:
        legacy applied -> status
        legacy notes   -> note
    """
    path: Path

    def _store(self) -> _NewHistoryStore:
        # client_id no está en el schema legacy, lo inferimos del path si se puede
        # fallback: "default"
        cid = "default"
        try:
            # .../data/clients/<client_id>/history.json
            parts = list(self.path.parts)
            if "clients" in parts:
                i = parts.index("clients")
                if i + 1 < len(parts):
                    cid = str(parts[i + 1])
        except Exception:
            pass
        return _NewHistoryStore(client_id=cid, path=self.path)

    # ------------- Legacy-like API -------------

    def load(self) -> Dict[str, Any]:
        """Devuelve el esquema NUEVO (no el legacy)."""
        return self._store().load()

    def save(self, data: Dict[str, Any]) -> None:
        """Guarda usando escritura atómica del store nuevo."""
        self._store().save(data)

    def upsert_period_plan(
        self,
        period_key: str,
        insight_ids: List[str],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Crea plan del periodo si no existe (no machaca feedback) usando store nuevo.
        """
        self._store().upsert_period_plan(period_key=period_key, insight_ids=insight_ids, meta=meta or {})

    def update_item(
        self,
        period_key: str,
        insight_id: str,
        applied: Optional[bool] = None,
        outcome: Optional[str] = None,
        notes: Optional[str] = None,
        kpi_before: Optional[float] = None,  # legacy: ignorado en v1
        kpi_after: Optional[float] = None,   # legacy: ignorado en v1
    ) -> None:
        """
        Traducción:
          applied True  -> status="done"
          applied False -> status="planned"
          outcome -> outcome (solo si status done, el store nuevo ya aplica guardrail)
          notes -> note
        """
        status = None
        if applied is not None:
            status = "done" if bool(applied) else "planned"

        # store nuevo tiene guardrails y admite create_if_missing
        self._store().update_item(
            period_key=period_key,
            insight_id=insight_id,
            status=status,
            outcome=outcome,
            note=notes,
            create_if_missing=True,
        )

    # ------------- Legacy scoring helpers (best-effort) -------------

    def repetition_penalty(self, insight_id: str, last_n: int = 2) -> float:
        """
        Penaliza si el insight se recomendó recientemente.
        Implementación best-effort sobre store nuevo:
          - mira últimos `last_n` periodos en recommendations
        """
        data = self._store().load()
        recs = data.get("recommendations") or []
        if not isinstance(recs, list):
            return 0.0
        recent = recs[-int(max(1, last_n)) :]
        hits = 0
        for r in recent:
            for it in (r.get("items") or []):
                if str(it.get("insight_id")) == str(insight_id):
                    hits += 1
        return min(0.2 * hits, 0.4)

    def outcome_bonus(self, insight_id: str, lookback: int = 6) -> float:
        """
        Bonus/malus según outcome reciente (best-effort):
          improved => +0.25
          not_improved => -0.20
        """
        data = self._store().load()
        recs = data.get("recommendations") or []
        if not isinstance(recs, list):
            return 0.0
        recent = recs[-int(max(1, lookback)) :]
        best = 0.0
        for r in recent:
            for it in (r.get("items") or []):
                if str(it.get("insight_id")) != str(insight_id):
                    continue
                out = str(it.get("outcome", "unknown"))
                if out == "improved":
                    best = max(best, 0.25)
                elif out == "not_improved":
                    best = min(best, -0.20)
        return best