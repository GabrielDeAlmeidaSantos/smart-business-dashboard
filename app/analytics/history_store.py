from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class HistoryStore:
    """JSON persistence for per-client plans and feedback.

    File schema (legacy v1, still valid):
    {
      "schema_version": 1,
      "client_id": "default",
      "recommendations": [
        {
          "period": "YYYY-MM-DD_YYYY-MM-DD",  # period_key basado en rango (vNext)
          "generated_at": "...",
          "meta": {...},
          "items": [
            {
              "insight_id": "uplift_worst_day_ticket",
              "status": "planned" | "done" | "skipped",
              "outcome": "unknown" | "improved" | "not_improved",
              "note": "",
              "updated_at": "..."
            }
          ]
        }
      ]
    }
    """
    client_id: str
    path: Path

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        self.ensure_parent()
        if not self.path.exists():
            return {"schema_version": 1, "client_id": self.client_id, "recommendations": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("History file root is not a dict.")
            data.setdefault("schema_version", 1)
            data.setdefault("client_id", self.client_id)
            data.setdefault("recommendations", [])
            return data
        except Exception:
            # Si está corrupto, arranca limpio (mejorable: backup)
            return {"schema_version": 1, "client_id": self.client_id, "recommendations": []}

    def save(self, data: Dict[str, Any]) -> None:
        self.ensure_parent()
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_period(self, period_key: str) -> Optional[Dict[str, Any]]:
        data = self.load()
        for rec in data.get("recommendations", []):
            if rec.get("period") == period_key:
                return rec
        return None

    def upsert_period_plan(self, period_key: str, insight_ids: list[str], meta: dict | None = None) -> None:
        """Create plan for period if missing; if exists, update ONLY if it has no items yet.

        This prevents overwriting feedback.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        existing = None
        for r in recs:
            if r.get("period") == period_key:
                existing = r
                break

        def _new_items():
            return [
                {
                    "insight_id": iid,
                    "status": "planned",
                    "outcome": "unknown",
                    "note": "",
                    "updated_at": _now_iso(),
                }
                for iid in insight_ids
            ]

        if existing is None:
            recs.append(
                {
                    "period": period_key,
                    "generated_at": _now_iso(),
                    "meta": meta or {},
                    "items": _new_items(),
                }
            )
            data["recommendations"] = recs
            self.save(data)
            return

        items = existing.get("items") or []
        if len(items) == 0:
            existing["items"] = _new_items()
            existing["meta"] = meta or existing.get("meta") or {}
            existing["generated_at"] = existing.get("generated_at") or _now_iso()
            self.save(data)

    def force_replace_period_plan(self, period_key: str, insight_ids: list[str], meta: dict | None = None) -> None:
        """Admin-only: replace plan list (resetea seguimiento)."""
        data = self.load()
        recs = data.get("recommendations") or []

        for r in recs:
            if r.get("period") == period_key:
                r["generated_at"] = _now_iso()
                r["meta"] = meta or {}
                r["items"] = [
                    {
                        "insight_id": iid,
                        "status": "planned",
                        "outcome": "unknown",
                        "note": "",
                        "updated_at": _now_iso(),
                    }
                    for iid in insight_ids
                ]
                self.save(data)
                return

        self.upsert_period_plan(period_key, insight_ids, meta=meta)

    def update_item(
        self,
        period_key: str,
        insight_id: str,
        status: str | None = None,
        outcome: str | None = None,
        note: str | None = None,
    ) -> bool:
        """Update a specific item inside a period.

        Returns:
            True if updated, False if period/item not found.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        for r in recs:
            if r.get("period") != period_key:
                continue

            items = r.get("items") or []
            for it in items:
                if it.get("insight_id") != insight_id:
                    continue

                if status is not None:
                    it["status"] = str(status)

                    # Guardrail: outcome solo tiene sentido si status == done
                    if str(status).strip().lower() != "done":
                        it["outcome"] = "unknown"

                if outcome is not None:
                    # Guardrail: no permitimos outcome != unknown si no está done
                    current_status = str(it.get("status", "planned")).strip().lower()
                    if current_status == "done":
                        it["outcome"] = str(outcome)
                    else:
                        it["outcome"] = "unknown"

                if note is not None:
                    it["note"] = str(note)

                it["updated_at"] = _now_iso()
                self.save(data)
                return True

            # Period existe pero no encontramos item
            return False

        # Period no existe
        return False