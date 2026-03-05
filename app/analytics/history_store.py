# app/analytics/history_store.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


_CLIENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_client_id(client_id: str) -> str:
    cid = str(client_id or "").strip()
    if not _CLIENT_ID_RE.match(cid):
        raise ValueError(
            "client_id inválido. Usa solo letras/números/guion/guion_bajo (1-64 chars)."
        )
    return cid


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Escritura atómica: write -> tmp -> replace. También crea .bak antes si existe."""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bak_path = path.with_suffix(path.suffix + ".bak")

    payload = json.dumps(data, ensure_ascii=False, indent=2)

    # 1) escribir tmp
    tmp_path.write_text(payload, encoding="utf-8")

    # 2) backup del actual si existe (best-effort)
    try:
        if path.exists():
            # os.replace sobreescribe si .bak ya existía (lo cual es OK)
            os.replace(str(path), str(bak_path))
    except Exception:
        # si falla el backup, no abortamos (pero la escritura atómica sigue)
        pass

    # 3) replace atómico (cross-platform overwrite)
    os.replace(str(tmp_path), str(path))


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Carga JSON; devuelve None si no puede."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


@dataclass
class HistoryStore:
    """JSON persistence for per-client plans and feedback (schema v1).

    Schema:
    {
      "schema_version": 1,
      "client_id": "default",
      "recommendations": [
        {
          "period": "YYYY-MM-DD_YYYY-MM-DD",
          "generated_at": "...",
          "meta": {...},
          "items": [
            {
              "insight_id": "...",
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

    def __post_init__(self) -> None:
        # Evita ids raros (y posibles traversal)
        self.client_id = _validate_client_id(self.client_id)

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty(self) -> Dict[str, Any]:
        return {"schema_version": 1, "client_id": self.client_id, "recommendations": []}

    def load(self) -> Dict[str, Any]:
        """Carga history.json; si está corrupto intenta .bak; si no, devuelve estructura vacía."""
        self.ensure_parent()

        if not self.path.exists():
            return self._empty()

        data = _safe_load_json(self.path)
        if data is None:
            # Intentar recuperación desde backup
            bak = self.path.with_suffix(self.path.suffix + ".bak")
            bak_data = _safe_load_json(bak) if bak.exists() else None
            if isinstance(bak_data, dict):
                data = bak_data
            else:
                # No reseteamos el archivo original aquí (para no perder evidencia).
                return self._empty()

        # Normalizar estructura
        data.setdefault("schema_version", 1)
        data.setdefault("client_id", self.client_id)
        data.setdefault("recommendations", [])

        # Asegurar tipo correcto
        if not isinstance(data.get("recommendations"), list):
            data["recommendations"] = []

        # Orden estable por generated_at (si falta, lo manda al final)
        try:
            def _k(r: dict) -> str:
                return str(r.get("generated_at") or "9999-12-31T23:59:59")
            data["recommendations"] = sorted(
                [r for r in data["recommendations"] if isinstance(r, dict)],
                key=_k,
            )
        except Exception:
            pass

        return data

    def save(self, data: Dict[str, Any]) -> None:
        """Guarda de forma atómica (+ backup)."""
        self.ensure_parent()
        _atomic_write_json(self.path, data)

    def get_period(self, period_key: str) -> Optional[Dict[str, Any]]:
        data = self.load()
        for rec in data.get("recommendations", []):
            if rec.get("period") == period_key:
                return rec
        return None

    def upsert_period_plan(self, period_key: str, insight_ids: list[str], meta: dict | None = None) -> None:
        """Crea plan si no existe; si existe, actualiza SOLO si no tiene items (no machaca feedback)."""
        data = self.load()
        recs = data.get("recommendations") or []

        existing: Optional[Dict[str, Any]] = None
        for r in recs:
            if r.get("period") == period_key:
                existing = r
                break

        def _new_items() -> list[Dict[str, Any]]:
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
        if not isinstance(items, list):
            items = []
            existing["items"] = items

        if len(items) == 0:
            existing["items"] = _new_items()
            existing["meta"] = meta or existing.get("meta") or {}
            existing["generated_at"] = existing.get("generated_at") or _now_iso()
            self.save(data)

    def force_replace_period_plan(self, period_key: str, insight_ids: list[str], meta: dict | None = None) -> None:
        """Admin-only: reemplaza plan (resetea seguimiento)."""
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
        *,
        create_if_missing: bool = False,
    ) -> bool:
        """Actualiza un item dentro de un periodo.

        Args:
            create_if_missing: si True y el item no existe pero el periodo sí, lo crea.

        Returns:
            True si actualiza/crea, False si no existe el periodo (o item y create_if_missing=False).
        """
        data = self.load()
        recs = data.get("recommendations") or []

        for r in recs:
            if r.get("period") != period_key:
                continue

            items = r.get("items") or []
            if not isinstance(items, list):
                items = []
                r["items"] = items

            target = None
            for it in items:
                if isinstance(it, dict) and it.get("insight_id") == insight_id:
                    target = it
                    break

            if target is None:
                if not create_if_missing:
                    return False
                target = {
                    "insight_id": insight_id,
                    "status": "planned",
                    "outcome": "unknown",
                    "note": "",
                    "updated_at": _now_iso(),
                }
                items.append(target)

            if status is not None:
                target["status"] = str(status)
                if str(status).strip().lower() != "done":
                    target["outcome"] = "unknown"

            if outcome is not None:
                current_status = str(target.get("status", "planned")).strip().lower()
                target["outcome"] = str(outcome) if current_status == "done" else "unknown"

            if note is not None:
                target["note"] = str(note)

            target["updated_at"] = _now_iso()
            self.save(data)
            return True

        return False