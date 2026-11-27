from __future__ import annotations

from datetime import datetime, timezone

from betdash.db.repository import Repository
from betdash.normalize.models import CanonicalEvent
from betdash.results.ingest import ingest_tennis_results_report


def test_ingest_report_counts(tmp_path):
    repo = Repository(str(tmp_path / 'bet.sqlite'))
    repo.upsert_canonical_event(CanonicalEvent(
        event_uid='e1', sport='tennis', league='ATP', participant_ids=('Djokovic N.','Alcaraz C.'),
        start_at=datetime(2026,3,1,12,0,tzinfo=timezone.utc), mapping_confidence=1.0,
    ))
    csv_path = tmp_path / 'r.csv'
    csv_path.write_text('Date,Winner,Loser,Score,Tournament\n2026-03-01,Djokovic N.,Alcaraz C.,6-4 6-4,ATP\n2026-03-01,Nadal R.,Alcaraz C.,6-4 6-4,ATP\n', encoding='utf-8')
    report = ingest_tennis_results_report(csv_path, repo)
    d = report.as_dict()
    assert d['parsed'] == 2
    assert d['stored'] == 1
    assert d['matched'] == 1
    assert d['unmatched'] == 1
    assert int(report) == 1
