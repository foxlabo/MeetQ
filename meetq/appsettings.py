"""UI-editable settings (F6).

Resolution order: value saved from the settings screen -> .env / environment default.
Model names are therefore always overridable through env, and never hardcoded here.
"""

from __future__ import annotations

from meetq.config import get_settings
from meetq.db.models import AppSetting
from meetq.db.session import db_session

KEYS = ("stt_model", "stt_live_model", "llm_model", "language", "glossary", "corrections")


def _defaults() -> dict[str, str]:
    s = get_settings()
    return {
        "stt_model": s.stt_model,
        "stt_live_model": s.stt_live_model,
        "llm_model": s.llm_model,
        "language": s.language,
        "glossary": "",
        "corrections": "",
    }


def all_settings() -> dict[str, str]:
    values = _defaults()
    try:
        with db_session() as db:
            for row in db.query(AppSetting).all():
                if row.key in values and row.value:
                    values[row.key] = row.value
    except Exception:  # pragma: no cover - DB not initialised yet
        pass
    return values


def get(key: str) -> str:
    return all_settings()[key]


def save(updates: dict[str, str]) -> dict[str, str]:
    with db_session() as db:
        for key, value in updates.items():
            if key not in KEYS:
                continue
            row = db.get(AppSetting, key)
            if row is None:
                row = AppSetting(key=key, value=value or "")
                db.add(row)
            else:
                row.value = value or ""
    return all_settings()
