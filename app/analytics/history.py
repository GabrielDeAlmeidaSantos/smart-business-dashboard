from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal


Outcome = Literal["unknown", "improved", "not_improved"]


@dataclass
class HistoryStore:
    """
    Historial por cliente.

    Estructura:
      {
        "recommendations": [
          {
            "period": "YYYY-MM",
            "items": [
              {
                "insight_id": "...",
                "applied": false,
                "outcome": "unknown|improved|not_improved",
                "notes": "",
                "kpi_before": null,
                "kpi_after": null
              }, ...
            ],
            "meta": {"generated_at": "...", ...}
          }, ...
        ]
      }
    """
    path: Path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"recommendations": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert_period_plan(
        self,
        period_key: str,
        insight_ids: list[str],
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Crea o reemplaza el plan del periodo.
        Mantiene los estados previos si el insight ya existía en ese periodo.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        existing = None
        for r in recs:
            if r.get("period") == period_key:
                existing = r
                break

        old_items_by_id = {}
        if existing:
            for it in (existing.get("items") or []):
                old_items_by_id[str(it.get("insight_id"))] = it

        new_items = []
        for iid in insight_ids:
            prev = old_items_by_id.get(iid, {})
            new_items.append(
                {
                    "insight_id": iid,
                    "applied": bool(prev.get("applied", False)),
                    "outcome": prev.get("outcome", "unknown"),
                    "notes": prev.get("notes", ""),
                    "kpi_before": prev.get("kpi_before", None),
                    "kpi_after": prev.get("kpi_after", None),
                }
            )

        payload = {
            "period": period_key,
            "items": new_items,
            "meta": meta or {},
        }

        if existing:
            # replace in place
            for i, r in enumerate(recs):
                if r.get("period") == period_key:
                    recs[i] = payload
                    break
        else:
            recs.append(payload)

        data["recommendations"] = recs
        self.save(data)

    def update_item(
        self,
        period_key: str,
        insight_id: str,
        applied: Optional[bool] = None,
        outcome: Optional[str] = None,
        notes: Optional[str] = None,
        kpi_before: Optional[float] = None,
        kpi_after: Optional[float] = None,
    ) -> None:
        """
        Actualiza seguimiento de 1 insight dentro de 1 periodo.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        for r in recs:
            if r.get("period") != period_key:
                continue
            items = r.get("items") or []
            for it in items:
                if str(it.get("insight_id")) != insight_id:
                    continue
                if applied is not None:
                    it["applied"] = bool(applied)
                if outcome is not None:
                    it["outcome"] = outcome
                if notes is not None:
                    it["notes"] = notes
                if kpi_before is not None:
                    it["kpi_before"] = float(kpi_before)
                if kpi_after is not None:
                    it["kpi_after"] = float(kpi_after)
                self.save(data)
                return

        # si no existe el periodo o el insight, no hace nada (evita corrupción)
        self.save(data)

    def repetition_penalty(self, insight_id: str, last_n: int = 2) -> float:
        """
        Penaliza si el insight se recomendó recientemente.
        """
        data = self.load()
        recs = (data.get("recommendations") or [])[-last_n:]
        hits = 0
        for r in recs:
            for it in (r.get("items") or []):
                if str(it.get("insight_id")) == insight_id:
                    hits += 1
        return min(0.2 * hits, 0.4)

    def outcome_bonus(self, insight_id: str, lookback: int = 6) -> float:
        """
        Bonus/malus según histórico:
          improved => +0.25
          not_improved => -0.20
        Usa las últimas `lookback` entradas.
        """
        data = self.load()
        recs = (data.get("recommendations") or [])[-lookback:]
        best = 0.0

        for r in recs:
            for it in (r.get("items") or []):
                if str(it.get("insight_id")) != insight_id:
                    continue
                outcome = str(it.get("outcome", "unknown"))
                if outcome == "improved":
                    best = max(best, 0.25)
                elif outcome == "not_improved":
                    best = max(best, -0.20)

        return best