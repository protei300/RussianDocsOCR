"""Service configuration.

Two tiers, split by who changes them and how often:

* **Here (env / ``.env``)** — secrets and things that cannot change without a
  restart: the PIN, the JWT secret, the default API key, the data directory.
* **The settings store** (``core/settings_schema.py`` + ``repositories/settings.py``)
  — operator-tunable knobs the worker re-reads every loop, so a change applies
  without bouncing the service.

Secrets never appear in the settings store and are never returned by any
endpoint. That split is the reason ``auth_pin``/``jwt_secret``/``default_api_key``
live in this file and nowhere else.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")

    # --- Auth ---------------------------------------------------------------
    #: Protects the *website* only. API endpoints authenticate with API keys.
    auth_pin: str = "1234"
    jwt_secret: str = "changeme-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480  # 8 h, one working day

    #: The bootstrap API key. Always present and never deletable — without it a
    #: restart (which wipes runtime-created keys) would leave the API with no
    #: way in at all.
    #:
    #: Left empty on purpose. A hardcoded fallback would mean every deployment
    #: that forgot to set this shares one publicly-known key, so instead a
    #: random one is generated at startup (see ``core.auth.resolve_default_key``)
    #: and logged. Set this env var when you need a stable key that survives
    #: restarts — integrations otherwise have to be re-pointed each time.
    default_api_key: str = ""

    # --- Storage ------------------------------------------------------------
    #: SQLAlchemy connection URL for the metadata store. When empty the service
    #: falls back to a temporary on-disk store that is wiped at every start —
    #: fine for a demo, useless for anything real, so startup says so loudly.
    #:
    #: Supported and tested::
    #:
    #:   mssql+pyodbc://user:pass@host:1433/db?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes
    #:   postgresql+psycopg://user:pass@host:5432/db
    #:
    #: Aliased explicitly so the env var reads unambiguously in a compose file.
    database_connectionstring: str = Field(
        default="", validation_alias="RUSSIANDOCS_DATABASE_CONNECTIONSTRING")

    #: Where uploaded originals, canvases and thumbnails live. Artifacts stay on
    #: the filesystem in both modes — multi-megabyte PNGs do not belong in a
    #: database row.
    data_dir: str = "data"

    #: Wipe ``data_dir`` at startup.
    #:
    #: Only meaningful in temporary mode. With a database configured the rows
    #: outlive the process, so wiping the images would leave every stored
    #: document pointing at a missing file — ``resolve_storage_mode()`` forces
    #: this off in that case. Note "no Docker volume" alone does NOT make the
    #: directory ephemeral: ``docker restart`` keeps the writable layer.
    data_wipe_on_start: bool = True
    max_upload_mb: int = 20

    #: Queue the anonymised documents from ``samples/`` when the store is empty,
    #: so the log demonstrates real results instead of showing nothing. Only
    #: ever repository samples — never user uploads. 0 disables seeding,
    #: a positive number caps how many are queued.
    seed_samples: int = 0  # 0 = all available; set negative to disable

    # --- Recognition --------------------------------------------------------
    compute_device: str = "auto"       # auto | cpu | gpu
    model_format: str = "ONNX"         # ONNX | OpenVINO
    ocr_mode: str = "accurate"         # accurate | fast  ('legacy' was removed in 3.0.0)
    pipeline_pool_size: int = 1
    #: Anonymised repository sample. Never point this at a real user document.
    warmup_image: str = ""

    # --- Worker -------------------------------------------------------------
    job_timeout_sec: int = 120
    max_retries: int = 2
    docconf: float = 0.5
    img_size: int = 1500

    # --- Ops ----------------------------------------------------------------
    log_level: str = "INFO"
    cors_allowed_origins: str = ""
    git_commit: str = "unknown"


@lru_cache
def get_settings() -> Settings:
    return Settings()
