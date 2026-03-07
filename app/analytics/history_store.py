from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# OBJETIVO DE ESTA VERSIÓN
# - Convertir el histórico en una pieza fuerte del producto.
# - Guardar mejor experimentos, revisiones y aprendizajes.
# - Dar mejor soporte a Owner, ranking y próximas acciones.
# - Dejar listo el sistema para responder bien a:
#     * qué se probó,
#     * qué pasó,
#     * qué hacemos ahora.
#
# CAMBIOS PRINCIPALES
# 1) Se añade una capa explícita de experimento dentro de cada item.
# 2) Se normaliza mejor el ciclo planned -> active -> done/reviewed.
# 3) Se crean helpers para estado actual, backlog y próximos focos.
# 4) El resumen histórico deja de ser solo sets y pasa a incluir señales útiles.
# 5) Se mantiene compatibilidad razonable con schema v2 anteriores.
# =============================================================================


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


_CLIENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ============================================================
# Estados y outcomes del MVP comercial
# ============================================================
VALID_STATUS = {
    "planned",         # planificada
    "active",          # en marcha / prueba viva
    "done",            # aplicada / cerrada a la espera de revisión final o ya revisada
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


# =============================================================================
# Validaciones básicas
# =============================================================================
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

    aliases = {
        "hecho": "done",
        "done": "done",
        "luego": "active",
        "active": "active",
        "saltar": "skipped",
        "skip": "skipped",
        "skipped": "skipped",
        "ya_lo_hacemos": "already_doing",
        "ya lo hacemos": "already_doing",
        "already_doing": "already_doing",
        "planned": "planned",
        "planificada": "planned",
    }
    out = aliases.get(value)
    return out if out in VALID_STATUS else None



def _normalize_outcome(outcome: str | None) -> Optional[str]:
    if outcome is None:
        return None
    value = str(outcome).strip().lower()

    aliases = {
        "unknown": "unknown",
        "improved": "improved",
        "positivo": "improved",
        "positive": "improved",
        "not_improved": "not_improved",
        "negative": "not_improved",
        "negativo": "not_improved",
        "no mejoró": "not_improved",
        "no mejoro": "not_improved",
        "inconclusive": "inconclusive",
        "inconcluso": "inconclusive",
        "neutral": "neutral",
        "neutro": "neutral",
    }
    out = aliases.get(value)
    return out if out in VALID_OUTCOME else None



def _parse_iso(dt_str: str | None) -> Optional[datetime]:
    try:
        if not dt_str:
            return None
        return datetime.fromisoformat(str(dt_str))
    except Exception:
        return None



def _days_from_now_iso(days: int) -> str:
    return (datetime.now() + timedelta(days=int(days))).isoformat(timespec="seconds")



def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


# =============================================================================
# IO seguro
# =============================================================================
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


# =============================================================================
# Store
# =============================================================================
@dataclass
class HistoryStore:
    """
    Persistencia JSON para seguimiento y aprendizaje del MVP.

    Schema v3 (lectura compatible con v2):

    {
      "schema_version": 3,
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
              "last_action_at": null,
              "last_outcome_at": null,
              "primary_metric_key": null,
              "primary_metric_label": null,
              "baseline_value": null,
              "observed_value": null,
              "delta_absolute": null,
              "delta_relative": null,
              "experiment": {
                "hypothesis": "",
                "where_to_apply": "",
                "success_rule_text": "",
                "review_window_label": "",
                "duration_days": null
              },
              "updated_at": "..."
            }
          ]
        }
      ]
    }

    Notas:
    - No hace falta que toda la app use ya todos los campos nuevos.
    - La idea es dejar el sistema listo para una UI de aprendizaje seria.
    """

    client_id: str
    path: Path

    def __post_init__(self) -> None:
        self.client_id = _validate_client_id(self.client_id)

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _empty(self) -> Dict[str, Any]:
        return {
            "schema_version": 3,
            "client_id": self.client_id,
            "recommendations": [],
        }

    # ============================================================
    # Carga / guardado
    # ============================================================
    def load(self) -> Dict[str, Any]:
        """
        Carga history.json; si está corrupto intenta .bak; si no, devuelve vacío.
        Además normaliza a schema interno consistente.
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

        data.setdefault("schema_version", 3)
        data.setdefault("client_id", self.client_id)
        data.setdefault("recommendations", [])

        if not isinstance(data.get("recommendations"), list):
            data["recommendations"] = []

        normalized_recs: list[dict] = []
        for rec in data["recommendations"]:
            if not isinstance(rec, dict):
                continue

            period = str(rec.get("period") or "")
            generated_at = str(rec.get("generated_at") or _now_iso())
            meta = rec.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}

            items = rec.get("items") or []
            if not isinstance(items, list):
                items = []

            normalized_items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                normalized_items.append(self._normalize_item(it))

            normalized_recs.append(
                {
                    "period": period,
                    "generated_at": generated_at,
                    "meta": meta,
                    "items": normalized_items,
                }
            )

        try:
            normalized_recs = sorted(
                normalized_recs,
                key=lambda r: str(r.get("generated_at") or "9999-12-31T23:59:59"),
            )
        except Exception:
            pass

        data["recommendations"] = normalized_recs
        data["schema_version"] = 3
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.ensure_parent()
        _atomic_write_json(self.path, data)

    # ============================================================
    # Normalización interna
    # ============================================================
    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza items antiguos/nuevos a una forma consistente.
        """
        status = _normalize_status(item.get("status")) or "planned"
        outcome = _normalize_outcome(item.get("outcome")) or "unknown"

        # Compatibilidad con antiguas representaciones malas
        if str(item.get("outcome") or "").strip().lower() == "already_doing":
            status = "already_doing"
            outcome = "unknown"

        experiment = item.get("experiment") or {}
        if not isinstance(experiment, dict):
            experiment = {}

        normalized = {
            "insight_id": str(item.get("insight_id") or ""),
            "status": status,
            "outcome": outcome,
            "note": str(item.get("note") or ""),
            "learning_note": str(item.get("learning_note") or ""),
            "started_at": item.get("started_at"),
            "review_due_at": item.get("review_due_at"),
            "reviewed_at": item.get("reviewed_at"),
            "last_action_at": item.get("last_action_at") or item.get("updated_at"),
            "last_outcome_at": item.get("last_outcome_at") or item.get("reviewed_at"),
            "primary_metric_key": item.get("primary_metric_key"),
            "primary_metric_label": item.get("primary_metric_label"),
            "baseline_value": _float_or_none(item.get("baseline_value")),
            "observed_value": _float_or_none(item.get("observed_value")),
            "delta_absolute": _float_or_none(item.get("delta_absolute")),
            "delta_relative": _float_or_none(item.get("delta_relative")),
            "experiment": {
                "hypothesis": str(experiment.get("hypothesis") or item.get("hypothesis") or ""),
                "where_to_apply": str(experiment.get("where_to_apply") or item.get("where_to_apply") or ""),
                "success_rule_text": str(experiment.get("success_rule_text") or item.get("success_rule_text") or ""),
                "review_window_label": str(experiment.get("review_window_label") or item.get("review_window_label") or ""),
                "duration_days": int(experiment.get("duration_days")) if str(experiment.get("duration_days") or "").isdigit() else None,
            },
            "updated_at": str(item.get("updated_at") or _now_iso()),
        }
        return normalized

    def _make_item(
        self,
        insight_id: str,
        *,
        primary_metric_key: str | None = None,
        primary_metric_label: str | None = None,
        hypothesis: str | None = None,
        where_to_apply: str | None = None,
        success_rule_text: str | None = None,
        review_window_label: str | None = None,
        duration_days: int | None = None,
    ) -> Dict[str, Any]:
        return {
            "insight_id": str(insight_id),
            "status": "planned",
            "outcome": "unknown",
            "note": "",
            "learning_note": "",
            "started_at": None,
            "review_due_at": _days_from_now_iso(duration_days or 14),
            "reviewed_at": None,
            "last_action_at": None,
            "last_outcome_at": None,
            "primary_metric_key": primary_metric_key,
            "primary_metric_label": primary_metric_label,
            "baseline_value": None,
            "observed_value": None,
            "delta_absolute": None,
            "delta_relative": None,
            "experiment": {
                "hypothesis": str(hypothesis or ""),
                "where_to_apply": str(where_to_apply or ""),
                "success_rule_text": str(success_rule_text or ""),
                "review_window_label": str(review_window_label or ""),
                "duration_days": int(duration_days) if duration_days is not None else None,
            },
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

    def list_periods(self) -> list[dict]:
        """Devuelve periodos ordenados del más reciente al más antiguo."""
        data = self.load()
        recs = [r for r in data.get("recommendations", []) if isinstance(r, dict)]
        recs.sort(key=lambda r: str(r.get("generated_at") or ""), reverse=True)
        return recs

    def upsert_period_plan(
        self,
        period_key: str,
        insight_ids: list[str],
        meta: dict | None = None,
        *,
        item_payloads: list[dict] | None = None,
    ) -> None:
        """
        Crea plan si no existe; si existe y no tiene items, lo rellena.

        item_payloads permite guardar desde el principio parte del contexto experimental.
        Esto deja el histórico mucho más útil para UI y aprendizaje.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        existing: Optional[Dict[str, Any]] = None
        for r in recs:
            if r.get("period") == period_key:
                existing = r
                break

        def _new_items() -> list[Dict[str, Any]]:
            out: list[Dict[str, Any]] = []
            if item_payloads:
                for p in item_payloads:
                    if not isinstance(p, dict):
                        continue
                    out.append(
                        self._make_item(
                            str(p.get("insight_id") or ""),
                            primary_metric_key=p.get("primary_metric_key"),
                            primary_metric_label=p.get("primary_metric_label"),
                            hypothesis=p.get("hypothesis"),
                            where_to_apply=p.get("where_to_apply"),
                            success_rule_text=p.get("success_rule_text"),
                            review_window_label=p.get("review_window_label"),
                            duration_days=p.get("duration_days"),
                        )
                    )
                if out:
                    return out
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
        *,
        item_payloads: list[dict] | None = None,
    ) -> None:
        """
        Admin-only: reemplaza plan y resetea seguimiento.
        """
        data = self.load()
        recs = data.get("recommendations") or []

        def _items() -> list[dict]:
            if item_payloads:
                built: list[dict] = []
                for p in item_payloads:
                    if not isinstance(p, dict):
                        continue
                    built.append(
                        self._make_item(
                            str(p.get("insight_id") or ""),
                            primary_metric_key=p.get("primary_metric_key"),
                            primary_metric_label=p.get("primary_metric_label"),
                            hypothesis=p.get("hypothesis"),
                            where_to_apply=p.get("where_to_apply"),
                            success_rule_text=p.get("success_rule_text"),
                            review_window_label=p.get("review_window_label"),
                            duration_days=p.get("duration_days"),
                        )
                    )
                if built:
                    return built
            return [self._make_item(iid) for iid in insight_ids]

        for r in recs:
            if r.get("period") == period_key:
                r["generated_at"] = _now_iso()
                r["meta"] = meta or {}
                r["items"] = _items()
                self.save(data)
                return

        self.upsert_period_plan(period_key, insight_ids, meta=meta, item_payloads=item_payloads)

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
        hypothesis: str | None = None,
        where_to_apply: str | None = None,
        success_rule_text: str | None = None,
        review_window_label: str | None = None,
        duration_days: int | None = None,
        create_if_missing: bool = False,
    ) -> bool:
        """
        Actualiza un item dentro de un periodo.

        Reglas:
        - status inválido -> se ignora
        - outcome inválido -> se ignora
        - si pasa a active y no tiene started_at -> se crea
        - si pasa a done y se informa outcome medido -> se puede sellar reviewed_at
        - el experimento vive en item['experiment']
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
                    hypothesis=hypothesis,
                    where_to_apply=where_to_apply,
                    success_rule_text=success_rule_text,
                    review_window_label=review_window_label,
                    duration_days=duration_days,
                )
                items.append(target)

            normalized_status = _normalize_status(status)
            normalized_outcome = _normalize_outcome(outcome)
            experiment = target.setdefault("experiment", {})
            if not isinstance(experiment, dict):
                experiment = {}
                target["experiment"] = experiment

            if normalized_status is not None:
                target["status"] = normalized_status
                target["last_action_at"] = _now_iso()

                if normalized_status == "active" and not target.get("started_at"):
                    target["started_at"] = _now_iso()

                if normalized_status == "done":
                    # si ya acabó y no tiene revisión programada, mantenemos review_due_at existente
                    if target.get("review_due_at") is None and duration_days is not None:
                        target["review_due_at"] = _days_from_now_iso(duration_days)

                if normalized_status in ("planned", "active", "skipped", "already_doing"):
                    if normalized_status != "done":
                        target["outcome"] = "unknown"

            if normalized_outcome is not None:
                current_status = str(target.get("status", "planned")).strip().lower()
                if current_status == "done":
                    target["outcome"] = normalized_outcome
                    target["last_outcome_at"] = _now_iso()
                    if reviewed_at is None and normalized_outcome != "unknown" and not target.get("reviewed_at"):
                        target["reviewed_at"] = _now_iso()
                else:
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

            if hypothesis is not None:
                experiment["hypothesis"] = str(hypothesis)

            if where_to_apply is not None:
                experiment["where_to_apply"] = str(where_to_apply)

            if success_rule_text is not None:
                experiment["success_rule_text"] = str(success_rule_text)

            if review_window_label is not None:
                experiment["review_window_label"] = str(review_window_label)

            if duration_days is not None:
                experiment["duration_days"] = int(duration_days)
                if target.get("review_due_at") is None:
                    target["review_due_at"] = _days_from_now_iso(duration_days)

            target["updated_at"] = _now_iso()
            self.save(data)
            return True

        return False

    # ============================================================
    # Helpers de estado / aprendizaje
    # ============================================================
    def get_latest_items_by_insight(self) -> dict[str, dict]:
        """Último estado conocido por insight_id en todo el histórico del cliente."""
        data = self.load()
        latest_by_insight: dict[str, dict] = {}

        for rec in data.get("recommendations", []):
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
                cur_dt = str(it.get("updated_at") or "")
                prev_dt = str((prev or {}).get("updated_at") or "")
                if prev is None or cur_dt >= prev_dt:
                    latest_by_insight[insight_id] = it

        return latest_by_insight

    def get_next_focus_items(self, limit: int = 3) -> list[dict]:
        """
        Devuelve acciones que parecen vivas o pendientes:
        - active
        - planned
        - already_doing
        Ordenadas por cercanía de revisión y actualización.
        """
        rows: list[dict] = []
        latest = self.get_latest_items_by_insight()

        for insight_id, it in latest.items():
            status = str(it.get("status") or "").strip().lower()
            if status not in {"active", "planned", "already_doing"}:
                continue

            due = str(it.get("review_due_at") or "")
            rows.append(
                {
                    "insight_id": insight_id,
                    "status": status,
                    "primary_metric_label": it.get("primary_metric_label"),
                    "review_due_at": due,
                    "updated_at": it.get("updated_at"),
                    "note": it.get("note"),
                    "learning_note": it.get("learning_note"),
                    "experiment": it.get("experiment") or {},
                }
            )

        rows.sort(key=lambda x: (str(x.get("review_due_at") or "9999"), str(x.get("updated_at") or "")), reverse=False)
        return rows[: max(1, int(limit))]

    # ============================================================
    # Resúmenes para rotación / ranking / UI
    # ============================================================
    def get_history_summary(
        self,
        client_id: str | None = None,
        period_key: str | None = None,
    ) -> Dict[str, Any]:
        """
        Resumen útil para:
        - apply_history_penalty()
        - ranking.py
        - recommender.py
        - bloques Owner/Admin

        Ahora incluye señales más ricas:
        - sets clásicos
        - contadores por outcome
        - backlog vivo
        - insights que funcionaron / no funcionaron / están pendientes
        """
        data = self.load()
        recs = data.get("recommendations") or []

        recent_done: set[str] = set()
        recent_seen: set[str] = set()
        already_doing: set[str] = set()
        recent_positive: set[str] = set()
        recent_negative: set[str] = set()
        recent_inconclusive: set[str] = set()
        active_now: set[str] = set()
        planned_now: set[str] = set()
        due_review: set[str] = set()

        latest_by_insight = self.get_latest_items_by_insight()
        now = datetime.now()

        for insight_id, it in latest_by_insight.items():
            recent_seen.add(insight_id)

            status = str(it.get("status") or "").strip().lower()
            outcome = str(it.get("outcome") or "").strip().lower()
            review_due_at = _parse_iso(it.get("review_due_at"))

            if status == "done":
                recent_done.add(insight_id)
            if status == "already_doing":
                already_doing.add(insight_id)
            if status == "active":
                active_now.add(insight_id)
            if status == "planned":
                planned_now.add(insight_id)
            if review_due_at is not None and review_due_at <= now and status in {"active", "done", "planned"}:
                due_review.add(insight_id)

            if outcome == "improved":
                recent_positive.add(insight_id)
            elif outcome == "not_improved":
                recent_negative.add(insight_id)
            elif outcome == "inconclusive":
                recent_inconclusive.add(insight_id)

        # Señal agregada básica para ranking y UI.
        total_periods = len([r for r in recs if isinstance(r, dict)])
        total_items = 0
        reviewed_items = 0
        improved_items = 0
        not_improved_items = 0
        inconclusive_items = 0

        for rec in recs:
            if not isinstance(rec, dict):
                continue
            items = rec.get("items") or []
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                total_items += 1
                outcome = str(it.get("outcome") or "unknown").strip().lower()
                if outcome in {"improved", "not_improved", "inconclusive", "neutral"} or it.get("reviewed_at"):
                    reviewed_items += 1
                if outcome == "improved":
                    improved_items += 1
                elif outcome == "not_improved":
                    not_improved_items += 1
                elif outcome == "inconclusive":
                    inconclusive_items += 1

        return {
            "recent_done": sorted(recent_done),
            "recent_seen": sorted(recent_seen),
            "already_doing": sorted(already_doing),
            "recent_positive": sorted(recent_positive),
            "recent_negative": sorted(recent_negative),
            "recent_inconclusive": sorted(recent_inconclusive),
            "active_now": sorted(active_now),
            "planned_now": sorted(planned_now),
            "due_review": sorted(due_review),
            "counts": {
                "periods": total_periods,
                "items": total_items,
                "reviewed_items": reviewed_items,
                "improved_items": improved_items,
                "not_improved_items": not_improved_items,
                "inconclusive_items": inconclusive_items,
            },
        }

    def get_recent_learnings(
        self,
        limit: int = 5,
    ) -> list[dict]:
        """
        Devuelve los últimos aprendizajes útiles para UI Owner/Admin.

        Ahora incluye también contexto experimental, lo que facilita mostrar:
        - qué se probó
        - dónde se aplicó
        - cuál era la hipótesis
        - qué pasó
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

                outcome = str(it.get("outcome") or "unknown").strip().lower()
                experiment = it.get("experiment") or {}
                if not isinstance(experiment, dict):
                    experiment = {}

                if not any(
                    [
                        it.get("learning_note"),
                        outcome in ("improved", "not_improved", "inconclusive", "neutral"),
                        it.get("reviewed_at"),
                        it.get("observed_value") is not None,
                    ]
                ):
                    continue

                rows.append(
                    {
                        "period": period,
                        "insight_id": it.get("insight_id"),
                        "status": it.get("status"),
                        "outcome": outcome,
                        "learning_note": it.get("learning_note"),
                        "note": it.get("note"),
                        "primary_metric_key": it.get("primary_metric_key"),
                        "primary_metric_label": it.get("primary_metric_label"),
                        "baseline_value": it.get("baseline_value"),
                        "observed_value": it.get("observed_value"),
                        "delta_absolute": it.get("delta_absolute"),
                        "delta_relative": it.get("delta_relative"),
                        "started_at": it.get("started_at"),
                        "review_due_at": it.get("review_due_at"),
                        "reviewed_at": it.get("reviewed_at"),
                        "updated_at": it.get("updated_at"),
                        "hypothesis": experiment.get("hypothesis"),
                        "where_to_apply": experiment.get("where_to_apply"),
                        "success_rule_text": experiment.get("success_rule_text"),
                        "review_window_label": experiment.get("review_window_label"),
                        "duration_days": experiment.get("duration_days"),
                    }
                )

        rows.sort(
            key=lambda x: str(x.get("reviewed_at") or x.get("updated_at") or ""),
            reverse=True,
        )
        return rows[: max(1, int(limit))]

    def get_action_story(self, limit: int = 10) -> list[dict]:
        """
        Timeline compacta para futuras vistas tipo:
        - qué se probó,
        - qué pasó,
        - qué hacemos ahora.
        """
        rows: list[dict] = []
        data = self.load()

        for rec in data.get("recommendations", []):
            if not isinstance(rec, dict):
                continue
            period = rec.get("period")
            generated_at = rec.get("generated_at")
            items = rec.get("items") or []
            if not isinstance(items, list):
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue
                experiment = it.get("experiment") or {}
                if not isinstance(experiment, dict):
                    experiment = {}

                rows.append(
                    {
                        "period": period,
                        "generated_at": generated_at,
                        "insight_id": it.get("insight_id"),
                        "status": it.get("status"),
                        "outcome": it.get("outcome"),
                        "started_at": it.get("started_at"),
                        "reviewed_at": it.get("reviewed_at"),
                        "updated_at": it.get("updated_at"),
                        "primary_metric_label": it.get("primary_metric_label"),
                        "learning_note": it.get("learning_note"),
                        "hypothesis": experiment.get("hypothesis"),
                        "where_to_apply": experiment.get("where_to_apply"),
                    }
                )

        rows.sort(key=lambda x: str(x.get("reviewed_at") or x.get("updated_at") or ""), reverse=True)
        return rows[: max(1, int(limit))]