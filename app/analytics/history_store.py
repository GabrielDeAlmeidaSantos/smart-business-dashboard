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

# ============================================================
# Estados y outcomes del MVP comercial
# ============================================================
VALID_STATUS = {
    "planned",         # planificada
    "active",          # en marcha / luego convertida en prueba viva
    "done",            # hecha / aplicada
    "skipped",         # saltada
    "already_doing",   # ya la hacen
}

VALID_OUTCOME = {
    "unknown",
    "improved",
    "not_improved",
    "inconclusive",
    "neutral",
}


def _validate_client_id(client_id: str) -> str:
    cid = str(client_id or "").strip()
    if not _CLIENT_ID_RE.match(cid):
        raise ValueError(
            "client_id inválido. Usa solo letras/números/guion/guion_bajo (1-64 chars)."
        )
    return cid


def _normalize_status(status: str | None) -> Optional[str]:
    if status is None:
        return None
    value = str(status).strip().lower()

    # Compatibilidad con etiquetas de UI más humanas
    aliases = {
        "hecho": "done",
        "done": "done",
        "luego": "active",
        "active": "active",
        "saltar": "skipped",
        "skipped": "skipped",
        "ya_lo_hacemos": "already_doing",
        "already_doing": "already_doing",
        "planned": "planned",
    }
    return aliases.get(value)


def _normalize_outcome(outcome: str | None) -> Optional[str]:
    if outcome is None:
        return None
    value = str(outcome).strip().lower()

    aliases = {
        "unknown": "unknown",
        "improved": "improved",
        "positivo": "improved",
        "not_improved": "not_improved",
        "negative": "not_improved",
        "negativo": "not_improved",
        "inconclusive": "inconclusive",
        "inconcluso": "inconclusive",
        "neutral": "neutral",
        "neutro": "neutral",
    }
    return aliases.get(value)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Escritura atómica:
    write -> tmp -> replace.
    También crea .bak si existe un archivo previo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    bak_path = path.with_suffix(path.suffix + ".bak")

    payload = json.dumps(data, ensure_ascii=False, indent=2)

    tmp_path.write_text(payload, encoding="utf-8")

    try:
        if path.exists():
            os.replace(str(path), str(bak_path))
    except Exception:
        pass

    os.replace(str(tmp_path), str(path))


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


@dataclass
class HistoryStore:
    """
    Persistencia JSON para seguimiento y aprendizaje del MVP.

    Schema v2 (compatible con lectura de v1):

    {
      "schema_version": 2,
      "client_id": "default",
      "recommendations": [
        {
          "period": "YYYY-MM-DD_YYYY-MM-DD",
          "generated_at": "...",
          "meta": {...},
          "items": [
            {
              "insight_id": "...",
              "status": "planned|active|done|skipped|already_doing",
              "outcome": "unknown|improved|not_improved|inconclusive|neutral",
              "note": "",
              "learning_note": "",
              "started_at": null,
              "review_due_at": null,
              "reviewed_at": null,
              "primary_metric_key": null,
              "primary_metric_label": null,
              "baseline_value": null,
              "observed_value": null,
              "delta_absolute": null,
              "delta_relative": null,
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
        self.client_id = _validate_client_id(self.client_id)

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty(self) -> Dict[str, Any]:
        return {
            "schema_version": 2,
            "client_id": self.client_id,
            "recommendations": [],
        }

    def load(self) -> Dict[str, Any]:
        """
        Carga history.json; si está corrupto intenta .bak; si no, devuelve vacío.
        Además hace normalización ligera de schema.
        """
        self.ensure_parent()

        if not self.path.exists():
            return self._empty()

        data = _safe_load_json(self.path)
        if data is None:
            bak = self.path.with_suffix(self.path.suffix + ".bak")
            bak_data = _safe_load_json(bak) if bak.exists() else None
            if isinstance(bak_data, dict):
                data = bak_data
            else:
                return self._empty()

        data.setdefault("schema_version", 2)
        data.setdefault("client_id", self.client_id)
        data.setdefault("recommendations", [])

        if not isinstance(data.get("recommendations"), list):
            data["recommendations"] = []

        # Normalización ligera de items antiguos
        for rec in data["recommendations"]:
            if not isinstance(rec, dict):
                continue
            rec.setdefault("period", "")
            rec.setdefault("generated_at", _now_iso())
            rec.setdefault("meta", {})
            rec.setdefault("items", [])

            if not isinstance(rec["items"], list):
                rec["items"] = []

            normalized_items = []
            for it in rec["items"]:
                if not isinstance(it, dict):
                    continue
                normalized_items.append(self._normalize_item(it))
            rec["items"] = normalized_items

        try:
            data["recommendations"] = sorted(
                [r for r in data["recommendations"] if isinstance(r, dict)],
                key=lambda r: str(r.get("generated_at") or "9999-12-31T23:59:59"),
            )
        except Exception:
            pass

        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.ensure_parent()
        _atomic_write_json(self.path, data)

    # ============================================================
    # Normalización interna
    # ============================================================
    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza items v1/v2 a una forma consistente.
        """
        status = _normalize_status(item.get("status")) or "planned"
        outcome = _normalize_outcome(item.get("outcome")) or "unknown"

        # already_doing debe vivir como status, no como outcome
        if str(item.get("outcome") or "").strip().lower() == "already_doing":
            status = "already_doing"
            outcome = "unknown"

        normalized = {
            "insight_id": str(item.get("insight_id") or ""),
            "status": status,
            "outcome": outcome,
            "note": str(item.get("note") or ""),
            "learning_note": str(item.get("learning_note") or ""),
            "started_at": item.get("started_at"),
            "review_due_at": item.get("review_due_at"),
            "reviewed_at": item.get("reviewed_at"),
            "primary_metric_key": item.get("primary_metric_key"),
            "primary_metric_label": item.get("primary_metric_label"),
            "baseline_value": item.get("baseline_value"),
            "observed_value": item.get("observed_value"),
            "delta_absolute": item.get("delta_absolute"),
            "delta_relative": item.get("delta_relative"),
            "updated_at": str(item.get("updated_at") or _now_iso()),
        }
        return normalized

    def _make_item(
        self,
        insight_id: str,
        *,
        primary_metric_key: str | None = None,
        primary_metric_label: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "insight_id": str(insight_id),
            "status": "planned",
            "outcome": "unknown",
            "note": "",
            "learning_note": "",
            "started_at": None,
            "review_due_at": None,
            "reviewed_at": None,
            "primary_metric_key": primary_metric_key,
            "primary_metric_label": primary_metric_label,
            "baseline_value": None,
            "observed_value": None,
            "delta_absolute": None,
            "delta_relative": None,
            "updated_at": _now_iso(),
        }

    # ============================================================
    # Lectura por periodo
    # ============================================================
    def get_period(self, period_key: str) -> Optional[Dict[str, Any]]:
        data = self.load()
        for rec in data.get("recommendations", []):
            if rec.get("period") == period_key:
                return rec
        return None

    def upsert_period_plan(
        self,
        period_key: str,
        insight_ids: list[str],
        meta: dict | None = None,
    ) -> None:
        """
        Crea plan si no existe; si existe y no tiene items, lo rellena.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        existing: Optional[Dict[str, Any]] = None
        for r in recs:
            if r.get("period") == period_key:
                existing = r
                break

        def _new_items() -> list[Dict[str, Any]]:
            return [self._make_item(iid) for iid in insight_ids]

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

    def force_replace_period_plan(
        self,
        period_key: str,
        insight_ids: list[str],
        meta: dict | None = None,
    ) -> None:
        """
        Admin-only: reemplaza plan y resetea seguimiento.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        for r in recs:
            if r.get("period") == period_key:
                r["generated_at"] = _now_iso()
                r["meta"] = meta or {}
                r["items"] = [self._make_item(iid) for iid in insight_ids]
                self.save(data)
                return

        self.upsert_period_plan(period_key, insight_ids, meta=meta)

    # ============================================================
    # Actualización de ítems
    # ============================================================
    def update_item(
        self,
        period_key: str,
        insight_id: str,
        status: str | None = None,
        outcome: str | None = None,
        note: str | None = None,
        learning_note: str | None = None,
        *,
        primary_metric_key: str | None = None,
        primary_metric_label: str | None = None,
        baseline_value: float | None = None,
        observed_value: float | None = None,
        delta_absolute: float | None = None,
        delta_relative: float | None = None,
        review_due_at: str | None = None,
        reviewed_at: str | None = None,
        create_if_missing: bool = False,
    ) -> bool:
        """
        Actualiza un item dentro de un periodo.

        Reglas:
        - status inválido -> se ignora
        - outcome inválido -> se ignora
        - si pasa a active y no tiene started_at -> se crea
        - si pasa a done/skipped/already_doing y no tiene reviewed_at -> se puede dejar tal cual
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
                target = self._make_item(
                    insight_id,
                    primary_metric_key=primary_metric_key,
                    primary_metric_label=primary_metric_label,
                )
                items.append(target)

            normalized_status = _normalize_status(status)
            normalized_outcome = _normalize_outcome(outcome)

            if normalized_status is not None:
                target["status"] = normalized_status

                if normalized_status == "active" and not target.get("started_at"):
                    target["started_at"] = _now_iso()

                if normalized_status in ("planned", "active", "skipped", "already_doing"):
                    # Mientras no haya resultado medido, outcome se resetea
                    if normalized_status != "done":
                        target["outcome"] = "unknown"

            if normalized_outcome is not None:
                current_status = str(target.get("status", "planned")).strip().lower()
                if current_status == "done":
                    target["outcome"] = normalized_outcome
                else:
                    # evitamos outcomes "medidos" en estados no finales
                    target["outcome"] = "unknown"

            if note is not None:
                target["note"] = str(note)

            if learning_note is not None:
                target["learning_note"] = str(learning_note)

            if primary_metric_key is not None:
                target["primary_metric_key"] = str(primary_metric_key)

            if primary_metric_label is not None:
                target["primary_metric_label"] = str(primary_metric_label)

            if baseline_value is not None:
                target["baseline_value"] = float(baseline_value)

            if observed_value is not None:
                target["observed_value"] = float(observed_value)

            if delta_absolute is not None:
                target["delta_absolute"] = float(delta_absolute)

            if delta_relative is not None:
                target["delta_relative"] = float(delta_relative)

            if review_due_at is not None:
                target["review_due_at"] = str(review_due_at)

            if reviewed_at is not None:
                target["reviewed_at"] = str(reviewed_at)

            target["updated_at"] = _now_iso()
            self.save(data)
            return True

        return False

    # ============================================================
    # Resúmenes para rotación / ranking / UI
    # ============================================================
    def get_history_summary(
        self,
        client_id: str | None = None,
        period_key: str | None = None,
    ) -> Dict[str, Any]:
        """
        Resumen compacto para:
        - apply_history_penalty()
        - recommender.py
        - bloques de aprendizaje en UI

        period_key es opcional; por ahora devolvemos resumen global simple
        sobre todo el histórico del cliente.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        recent_done: set[str] = set()
        recent_seen: set[str] = set()
        already_doing: set[str] = set()
        recent_positive: set[str] = set()
        recent_negative: set[str] = set()
        recent_inconclusive: set[str] = set()

        latest_by_insight: dict[str, dict] = {}

        for rec in recs:
            if not isinstance(rec, dict):
                continue

            items = rec.get("items") or []
            if not isinstance(items, list):
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue
                insight_id = str(it.get("insight_id") or "").strip()
                if not insight_id:
                    continue

                prev = latest_by_insight.get(insight_id)
                current_dt = it.get("updated_at") or ""
                prev_dt = (prev or {}).get("updated_at") or ""

                if prev is None or str(current_dt) >= str(prev_dt):
                    latest_by_insight[insight_id] = it

        for insight_id, it in latest_by_insight.items():
            recent_seen.add(insight_id)

            status = str(it.get("status") or "").strip().lower()
            outcome = str(it.get("outcome") or "").strip().lower()

            if status == "done":
                recent_done.add(insight_id)
            if status == "already_doing":
                already_doing.add(insight_id)

            if outcome == "improved":
                recent_positive.add(insight_id)
            elif outcome == "not_improved":
                recent_negative.add(insight_id)
            elif outcome == "inconclusive":
                recent_inconclusive.add(insight_id)

        return {
            "recent_done": sorted(recent_done),
            "recent_seen": sorted(recent_seen),
            "already_doing": sorted(already_doing),
            "recent_positive": sorted(recent_positive),
            "recent_negative": sorted(recent_negative),
            "recent_inconclusive": sorted(recent_inconclusive),
        }

    def get_recent_learnings(
        self,
        limit: int = 5,
    ) -> list[dict]:
        """
        Devuelve los últimos aprendizajes útiles para UI Owner/Admin.
        """
        data = self.load()
        rows: list[dict] = []

        for rec in data.get("recommendations", []):
            if not isinstance(rec, dict):
                continue

            period = rec.get("period")
            items = rec.get("items") or []
            if not isinstance(items, list):
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue

                if not any(
                    [
                        it.get("learning_note"),
                        it.get("outcome") in ("improved", "not_improved", "inconclusive"),
                        it.get("reviewed_at"),
                    ]
                ):
                    continue

                rows.append(
                    {
                        "period": period,
                        "insight_id": it.get("insight_id"),
                        "status": it.get("status"),
                        "outcome": it.get("outcome"),
                        "learning_note": it.get("learning_note"),
                        "primary_metric_key": it.get("primary_metric_key"),
                        "primary_metric_label": it.get("primary_metric_label"),
                        "baseline_value": it.get("baseline_value"),
                        "observed_value": it.get("observed_value"),
                        "delta_absolute": it.get("delta_absolute"),
                        "delta_relative": it.get("delta_relative"),
                        "reviewed_at": it.get("reviewed_at"),
                        "updated_at": it.get("updated_at"),
                    }
                )

        rows.sort(key=lambda x: str(x.get("reviewed_at") or x.get("updated_at") or ""), reverse=True)
        return rows[: max(1, int(limit))]