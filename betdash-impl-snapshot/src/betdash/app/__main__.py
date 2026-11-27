"""CLI entrypoint for BetDash app.

`python -m betdash.app --check` prints worker/adapter status and exits 0.
`python -m betdash.app --tray` starts the tray icon controller.
`python -m betdash.app --ingest-tennis /path/to/file_or_dir.csv` imports tennis-data.co.uk CSV results.
"""

from __future__ import annotations

import argparse
import json

from betdash.app.dashboard import StreamlitRuntime
from betdash.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="betdash.app")
    parser.add_argument("--check", action="store_true", help="Print worker/adapter status and exit 0")
    parser.add_argument("--tray", action="store_true", help="Start tray icon controller (blocks until quit)")
    parser.add_argument(
        "--no-autostart-dashboard",
        action="store_true",
        help="With --tray, do not start dashboard subprocess automatically",
    )
    parser.add_argument(
        "--ingest-tennis",
        metavar="PATH",
        help="Import tennis-data.co.uk style CSV/XLSX file or directory into results table",
    )
    parser.add_argument(
        "--create-missing-events",
        action="store_true",
        help="When ingesting results, create minimal canonical events for unmatched rows (enables historical training)",
    )
    args = parser.parse_args(argv)

    if args.check:
        runtime = StreamlitRuntime()
        # Smoke check should not require full browser/long-running services.
        payload = runtime.check()
        print(json.dumps(payload, default=str, ensure_ascii=False, indent=2))
        return 0

    if args.ingest_tennis:
        from betdash.db.repository import Repository
        from betdash.results.ingest import ingest_tennis_results_report

        settings = Settings()
        repo = Repository(settings.db_path)
        report = ingest_tennis_results_report(
            args.ingest_tennis, repo, create_missing_events=bool(args.create_missing_events)
        )
        payload = report.as_dict()
        payload["db_path"] = settings.db_path
        payload["create_missing_events"] = bool(args.create_missing_events)
        print(json.dumps(payload, default=str, ensure_ascii=False, indent=2))
        return 0

    if args.tray:
        from betdash.app.tray import TrayController

        settings = Settings()
        controller = TrayController(
            settings=settings,
            autostart_dashboard=not args.no_autostart_dashboard,
            workdir=None,
        )
        controller.run()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
