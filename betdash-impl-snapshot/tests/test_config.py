from betdash.config import Settings


def test_settings_defaults():
    settings = Settings()

    assert settings.poll_seconds == 90
    assert settings.enable_catboost is False
    assert set(settings.bookmaker_hosts) == {
        "pari",
        "fonbet",
        "bettery",
        "marathon",
        "betboom",
        "winline",
    }
    assert settings.db_path
    assert settings.betboom_line_path
    assert settings.checkpoint_dir
    assert settings.max_daily_loss == float("inf")


def test_settings_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("POLL_SECONDS", "30")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "custom.sqlite"))
    monkeypatch.setenv("ENABLE_CATBOOST", "true")

    settings = Settings()

    assert settings.poll_seconds == 30
    assert settings.db_path == str(tmp_path / "custom.sqlite")
    assert settings.enable_catboost is True
