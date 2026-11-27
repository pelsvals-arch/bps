"""Tennis result ingestion from tennis-data.co.uk-style CSV/XLSX."""

from __future__ import annotations

import csv
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from betdash.db.repository import Repository
from betdash.results.resolve import resolve_result


@dataclass(frozen=True, slots=True)
class ResultRow:
    participant_a: str
    participant_b: str
    date: str
    score: str
    league: str
    source: str = "tennis-data"


@dataclass(slots=True)
class IngestReport:
    source: str = "tennis-data"
    files: list[str] = field(default_factory=list)
    total_rows: int = 0
    parsed: int = 0
    matched: int = 0
    stored: int = 0
    duplicates: int = 0
    unmatched: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "files": self.files,
            "total_rows": self.total_rows,
            "parsed": self.parsed,
            "matched": self.matched,
            "stored": self.stored,
            "duplicates": self.duplicates,
            "unmatched": self.unmatched,
            "errors": self.errors,
        }

    def __int__(self) -> int:  # backward compat
        return self.stored


def _get(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
        for k, v in row.items():
            if k.strip().lower() == key.strip().lower() and v is not None:
                value = str(v).strip()
                if value:
                    return value
    return ""


_XL_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _excel_serial_to_date(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text
    if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", text):
        return text  # resolve_result handles dd/mm/yyyy
    try:
        n = float(text)
    except ValueError:
        return text
    if n < 20000 or n > 60000:  # unlikely to be excel date
        return text
    # Excel 1900 date system; 25569 days between 1899-12-30 and 1970-01-01
    dt = datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=n)
    return dt.date().isoformat()


def _xlsx_shared_strings(z: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in z.namelist():
        return []
    root = ET.fromstring(z.read(name))
    shared: list[str] = []
    for si in root.findall("m:si", _XL_NS):
        texts = [t.text or "" for t in si.findall(".//m:t", _XL_NS)]
        shared.append("".join(texts))
    return shared


def _xlsx_rows(path: str | Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as z:
        shared = _xlsx_shared_strings(z)
        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in z.namelist():
            sheets = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet")]
            if not sheets:
                return []
            sheet_name = sorted(sheets)[0]
        root = ET.fromstring(z.read(sheet_name))
        rows: list[dict[str, str]] = []
        headers: list[str] | None = None
        for row in root.findall(".//m:sheetData/m:row", _XL_NS):
            vals: dict[int, str] = {}
            for c in row.findall("m:c", _XL_NS):
                ref = c.attrib.get("r", "A1")
                m = re.match(r"([A-Z]+)", ref)
                if not m:
                    continue
                col_letters = m.group(1)
                col = 0
                for ch in col_letters:
                    col = col * 26 + (ord(ch) - 64)
                col -= 1
                t = c.attrib.get("t")
                v = c.find("m:v", _XL_NS)
                is_elem = c.find("m:is", _XL_NS)
                if v is None and is_elem is None:
                    continue
                if v is not None:
                    val = v.text or ""
                else:
                    texts = [tn.text or "" for tn in is_elem.findall(".//m:t", _XL_NS)]
                    val = "".join(texts)
                if t == "s":
                    try:
                        val = shared[int(val)]
                    except Exception:
                        pass
                vals[col] = str(val).strip()
            if not vals:
                continue
            maxc = max(vals)
            row_vals = [vals.get(i, "") for i in range(maxc + 1)]
            if headers is None:
                headers = [str(h).strip() for h in row_vals]
                continue
            rec = {headers[i]: row_vals[i] for i in range(min(len(headers), len(row_vals)))}
            rows.append(rec)
        return rows


def _normalize_date(raw: str) -> str:
    return _excel_serial_to_date(raw)


def parse_tennis_results(path: str | Path) -> list[ResultRow]:
    """Parse tennis-data.co.uk CSV/XLSX columns into ResultRow records."""

    p = Path(path)
    rows: list[ResultRow] = []
    if p.suffix.lower() == ".xlsx":
        raw_rows = _xlsx_rows(p)
    else:
        with p.open(newline="", encoding="utf-8-sig") as handle:
            raw_rows = list(csv.DictReader(handle))
    for raw in raw_rows:
        date = _normalize_date(_get(raw, "Date"))
        winner = _get(raw, "Winner")
        loser = _get(raw, "Loser")
        score = _get(raw, "Score")
        wsets = _get(raw, "Wsets", "WSet", "W_sets")
        lsets = _get(raw, "Lsets", "LSet", "L_sets")
        comment = _get(raw, "Comment")
        if not score and wsets and lsets:
            score = f"{wsets}-{lsets}"
        league = _get(raw, "Tournament", "League", "Series")
        if not (date and winner and loser):
            continue
        # Skip rows with no score/sets and no decisive comment; they can't train.
        if not score and not comment:
            continue
        rows.append(
            ResultRow(
                participant_a=winner,
                participant_b=loser,
                date=date,
                score=score,
                league=league,
            )
        )
    return rows


def _iter_result_files(path: str | Path) -> list[Path]:
    p = Path(path)
    exts = {".csv", ".xlsx"}
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])
    return [p]


def ingest_tennis_results_report(
    path: str | Path, repo: Repository, registry: object | None = None, *, create_missing_events: bool = False
) -> IngestReport:
    """Parse, resolve, and store accepted tennis results. Returns detailed report.

    If create_missing_events=True, unmatched tennis-data rows create a minimal
    canonical event (sport=tennis, name-based participants) so historical
    results can train Elo/Glicko baseline even when odds snapshots are absent.
    """
    from datetime import datetime, timezone
    from betdash.normalize.registry import normalize_name
    from betdash.normalize.models import CanonicalEvent
    import hashlib

    report = IngestReport()
    files = _iter_result_files(path)
    report.files = [str(f) for f in files]
    events = repo.events_for_resolution()
    if not events and not create_missing_events:
        report.errors.append("no_canonical_events: ingest cannot resolve results without canonical events")
        # still parse to give feedback
    for file_path in files:
        try:
            parsed = parse_tennis_results(file_path)
        except Exception as e:
            report.errors.append(f"{file_path}: parse_error: {e}")
            continue
        if Path(file_path).suffix.lower() == ".xlsx":
            report.total_rows += len(parsed)
        else:
            try:
                with Path(file_path).open(newline="", encoding="utf-8-sig") as handle:
                    report.total_rows += sum(1 for _ in csv.DictReader(handle))
            except Exception:
                report.total_rows += len(parsed)
        report.parsed += len(parsed)
        if not events and not create_missing_events:
            report.unmatched += len(parsed)
            continue
        for result in parsed:
            try:
                event_uid = resolve_result(result, events, registry=registry) if events else None
            except Exception as e:
                report.errors.append(f"{file_path}: resolve_error: {e}")
                continue
            if event_uid is None and create_missing_events:
                try:
                    # deterministic historical event for training
                    dt = None
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
                        try:
                            dt = datetime.strptime(str(result.date).strip(), fmt).replace(tzinfo=timezone.utc)
                            break
                        except Exception:
                            pass
                    if dt is None:
                        dt = datetime.fromisoformat(str(result.date).strip().replace("Z","+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    p1 = normalize_name(result.participant_a)
                    p2 = normalize_name(result.participant_b)
                    ids = tuple(sorted([f"name:{p1}", f"name:{p2}"]))
                    uid_src = f"tennis|{result.league}|{ids[0]}|{ids[1]}|{dt.date().isoformat()}"
                    event_uid = hashlib.sha1(uid_src.encode("utf-8")).hexdigest()
                    ce = CanonicalEvent(
                        event_uid=event_uid,
                        sport="tennis",
                        league=result.league or "tennis",
                        participant_ids=ids,
                        start_at=dt,
                        mapping_confidence=0.7,
                    )
                    repo.upsert_canonical_event(ce)
                    events.append(ce)
                except Exception as e:
                    report.errors.append(f"{file_path}: create_event_error: {e}")
                    event_uid = None
            if event_uid is None:
                report.unmatched += 1
                continue
            report.matched += 1
            try:
                if repo.save_result(event_uid=event_uid, result=result):
                    report.stored += 1
                else:
                    report.duplicates += 1
            except Exception as e:
                report.errors.append(f"{file_path}: store_error: {e}")
    return report


def ingest_tennis_results(
    path: str | Path, repo: Repository, registry: object | None = None
) -> int:
    """Parse, resolve, and store accepted tennis results. Returns stored count."""
    return int(ingest_tennis_results_report(path, repo, registry=registry))

