# Bet Dashboard

Local, personal betting-research dashboard. It collects public **pre-match**
lines from six bookmakers, stores every observation as an immutable SQLite
snapshot, and uses walk-forward validation to compare model probabilities with
the market. Nothing here places bets, automates accounts or runs a public
server.

## Data sources and licenses

The project reads only public, unauthenticated pages and APIs. Bookmaker
interfaces are undocumented and may change at any time; each adapter records
its own health instead of failing the dashboard.

| Source | Used for | Terms |
| --- | --- | --- |
| Pari (`pari.ru`) | Public pre-match line | Public site data; personal, non-commercial research only. |
| Fonbet (`fon.bet`) | Public pre-match line | Public site data; personal, non-commercial research only. |
| Bettery (`bettery.ru`) | Public pre-match line | Public site data; personal, non-commercial research only. |
| Marathon (`marathonbet.ru`) | Public SSR line | Public site data; personal, non-commercial research only. |
| BetBoom (`betboom.ru`) | Public `siteapi` line | Public site data; personal, non-commercial research only. |
| Winline (`winline.ru`) | Public WebSocket line | Public site data; personal, non-commercial research only. |
| [tennis-data.co.uk](http://www.tennis-data.co.uk/) | Historical tennis results and odds | **Non-commercial use with attribution**; do not redistribute. |
| [tennis-sackmann-archive](https://github.com/JeffSackmann/tennis_sackmann) | Historical tennis results | **Non-commercial use with attribution**; no redistribution. |
| [OpenDota](https://www.opendota.com/) | Dota 2 results | OpenDota terms; attribution required. |
| [nba_api](https://github.com/swar/nba_api) | NBA results | Public NBA data; follow NBA terms. |
| Kaggle datasets | Optional CS2/basketball history | Each dataset's own license; check before use. |

The two tennis archives are explicitly non-commercial and require
attribution. Their data must not be redistributed, and the derived database
stays on the local machine.

## Layout

```
src/betdash/
  config.py          runtime settings (pydantic-settings)
  db/schema.sql      SQLite schema, WAL mode
  db/repository.py   init_db and append-only Repository
tests/
```

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pytest -q
```

## Running the dashboard

Smoke check (no long-running services):

```bash
PYTHONPATH=src .venv/bin/python -m betdash.app --check
```

Run dashboard directly:

```bash
PYTHONPATH=src .venv/bin/python -m streamlit run src/betdash/app/dashboard_page.py
```

Run tray controller (recommended, so you don't forget to stop dashboard):

```bash
PYTHONPATH=src .venv/bin/python -m betdash.app --tray
```

Tray menu: Start/Stop dashboard, Open in browser, Show status, Quit tray (stops dashboard). Dashboard subprocess is controlled by tray; workers live inside dashboard process to avoid duplicates.

The SQLite database is the source of truth; `odds_snapshots` is append-only
and deduplicated on `(event_uid, bookmaker, market, outcome, observed_at,
adapter_version)`.
