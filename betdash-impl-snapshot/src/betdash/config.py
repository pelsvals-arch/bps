"""Runtime configuration loaded from environment variables and ``.env``."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BOOKMAKER_HOSTS: dict[str, str] = {
    "pari": "https://pari.ru",
    "fonbet": "https://fon.bet",
    "bettery": "https://bettery.ru",
    "marathon": "https://new.marathonbet.ru",
    "betboom": "https://betboom.ru",
    "winline": "https://winline.ru",
}


class Settings(BaseSettings):
    """Application settings.

    Every field can be overridden with an environment variable of the same
    name in upper case (for example ``POLL_SECONDS=30``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: str = "data/betdash.sqlite"
    poll_seconds: int = 90
    bookmaker_hosts: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_BOOKMAKER_HOSTS)
    )
    betboom_line_path: str = "/siteapi/line/line"
    checkpoint_dir: str = "data/checkpoints"
    enable_catboost: bool = False
    max_daily_loss: float = float("inf")
    max_predictions: int = 2000
    tennis_prediction_horizon_hours: int = 72
