"""Application settings. Every model name is overridable via .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- secrets -------------------------------------------------------
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_timeout: float = Field(default=600.0, alias="OPENAI_TIMEOUT")

    # --- models (.env or the settings page override these) --------------
    stt_model: str = Field(default="gpt-transcribe", alias="MEETQ_STT_MODEL")
    stt_live_model: str = Field(default="gpt-live-transcribe", alias="MEETQ_STT_LIVE_MODEL")
    llm_model: str = Field(default="gpt-6-astra", alias="MEETQ_LLM_MODEL")

    # --- behaviour ------------------------------------------------------
    language: str = Field(default="ja", alias="MEETQ_LANGUAGE")
    llm_provider: str = Field(default="openai", alias="MEETQ_LLM_PROVIDER")
    host: str = Field(default="127.0.0.1", alias="MEETQ_HOST")
    port: int = Field(default=8765, alias="MEETQ_PORT")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="MEETQ_DATA_DIR")

    stt_concurrency: int = Field(default=3, alias="MEETQ_STT_CONCURRENCY")
    stt_max_retries: int = Field(default=5, alias="MEETQ_STT_MAX_RETRIES")
    chunk_target_sec: float = Field(default=20.0, alias="MEETQ_CHUNK_TARGET_SEC")
    chunk_min_sec: float = Field(default=10.0, alias="MEETQ_CHUNK_MIN_SEC")
    chunk_max_sec: float = Field(default=60.0, alias="MEETQ_CHUNK_MAX_SEC")

    # --- Phase 2: live transcription + rolling question generation ------
    live_sample_rate: int = Field(default=24000, alias="MEETQ_LIVE_SAMPLE_RATE")
    live_turn_min_sec: float = Field(default=2.0, alias="MEETQ_LIVE_TURN_MIN_SEC")
    live_turn_max_sec: float = Field(default=5.0, alias="MEETQ_LIVE_TURN_MAX_SEC")
    live_delay: str = Field(default="low", alias="MEETQ_LIVE_DELAY")
    rolling_interval_sec: float = Field(default=90.0, alias="MEETQ_ROLLING_INTERVAL_SEC")
    rolling_window_sec: float = Field(default=300.0, alias="MEETQ_ROLLING_WINDOW_SEC")

    @property
    def key_configured(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def db_path(self) -> Path:
        return self.data_dir / "meetq.sqlite3"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    return settings
