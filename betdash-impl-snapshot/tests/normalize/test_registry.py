"""Tests for the canonical event contract, alias registry and market mapping."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from betdash.normalize import (
    CanonicalEvent,
    EntityRegistry,
    MarketMapping,
    RawOdd,
    canon_market,
    canon_market_mapping,
    normalize_event,
    normalize_name,
)

FIXTURE = "tests/normalize/fixtures/aliases.json"
NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
BOOKMAKERS = ("winline", "pari", "fonbet", "marathon", "betboom", "bettery")
CANONICAL_LABELS = {
    "match_winner": "П1",
    "set_handicap": "Фора по сетам",
    "total_games": "Тотал геймов",
    "map_winner": "Победа по картам",
    "total_maps": "Тотал карт",
    "spread": "Фора",
    "total_points": "Тотал очков",
}


def make_raw(**overrides) -> RawOdd:
    values = {
        "bookmaker": "pari",
        "source_event_id": "123",
        "sport": "tennis",
        "league": "ATP",
        "participants": ("Djokovic N.", "Alcaraz C."),
        "market": "П1",
        "outcome": "p1",
        "odds_decimal": 1.8,
        "observed_at": NOW,
        "is_live": False,
    }
    values.update(overrides)
    return RawOdd(**values)


def registry() -> EntityRegistry:
    return EntityRegistry.from_json(FIXTURE)


# --- entity normalization -------------------------------------------------


def test_normalize_event_resolves_aliases():
    reg = registry()
    raw = RawOdd(
        bookmaker="pari",
        source_event_id="123",
        sport="tennis",
        league="ATP",
        participants=("Djokovic N.", "Alcaraz C."),
        market="П1",
        outcome="p1",
        odds_decimal=1.8,
        observed_at=NOW,
        is_live=False,
    )
    ev = normalize_event(raw, reg)
    assert ev.participant_ids == ("atp:djokovic", "atp:alcaraz")
    assert ev.mapping_confidence >= 0.8


def test_ambiguous_mapping_is_low_confidence():
    reg = registry()
    raw = make_raw(participants=("Ivanov", "Petrov"))
    assert normalize_event(raw, reg).mapping_confidence < 0.5


def test_unknown_mapping_uses_deterministic_name_confidence():
    reg = registry()
    ev = normalize_event(make_raw(participants=("Nobody", "Someone")), reg)
    assert ev.mapping_confidence == 0.6


def test_one_unknown_participant_uses_mixed_confidence():
    reg = registry()
    ev = normalize_event(make_raw(participants=("Djokovic N.", "Someone")), reg)
    assert ev.mapping_confidence == 0.6


def test_ambiguous_participants_get_provisional_ids():
    reg = registry()
    ev = normalize_event(make_raw(participants=("Ivanov", "Petrov")), reg)
    assert ev.participant_ids == ("ambiguous:ivanov", "ambiguous:petrov")


def test_unknown_participants_get_deterministic_name_ids():
    reg = registry()
    ev = normalize_event(make_raw(participants=("Nobody", "Someone")), reg)
    assert ev.participant_ids == ("name:nobody", "name:someone")


def test_canonical_event_keeps_observation_time_as_start():
    ev = normalize_event(make_raw(), registry())
    assert ev.start_at == NOW


def test_normalize_event_propagates_explicit_start_at():
    start = NOW + timedelta(days=2)
    ev = normalize_event(make_raw(start_at=start), registry())
    assert ev.start_at == start


def test_event_uid_uses_start_at_date():
    reg = registry()
    first = normalize_event(make_raw(start_at=NOW + timedelta(days=1)), reg)
    second = normalize_event(make_raw(start_at=NOW + timedelta(days=2)), reg)
    assert first.event_uid != second.event_uid


def test_event_uid_without_start_at_matches_explicit_observed_at():
    reg = registry()
    implicit = normalize_event(make_raw(), reg)
    explicit = normalize_event(make_raw(start_at=NOW), reg)
    assert implicit.start_at == NOW
    assert implicit.event_uid == explicit.event_uid


def test_event_uid_uses_utc_date_of_start_at():
    reg = registry()
    msk = timezone(timedelta(hours=3))
    # 2026-03-02 01:00 MSK is 2026-03-01 22:00 UTC.
    local = make_raw(start_at=datetime(2026, 3, 2, 1, 0, tzinfo=msk))
    utc = make_raw(start_at=datetime(2026, 3, 1, 22, 0, tzinfo=timezone.utc))
    assert normalize_event(local, reg).event_uid == normalize_event(utc, reg).event_uid


def test_canonical_event_keeps_raw_league_display():
    ev = normalize_event(make_raw(league="ATP"), registry())
    assert ev.league == "ATP"
    assert ev.sport == "tennis"


def test_normalize_event_is_case_insensitive_for_sport():
    ev = normalize_event(make_raw(sport="Tennis"), registry())
    assert ev.sport == "tennis"


# --- deterministic event ids ---------------------------------------------


def test_event_uid_is_sha1_hex():
    uid = normalize_event(make_raw(), registry()).event_uid
    assert len(uid) == 40
    int(uid, 16)


def test_event_uid_is_stable_across_books_and_participant_order():
    reg = registry()
    first = normalize_event(make_raw(bookmaker="pari"), reg)
    second = normalize_event(
        make_raw(
            bookmaker="winline",
            source_event_id="999",
            participants=("Alcaraz C.", "Djokovic N."),
        ),
        reg,
    )
    assert first.event_uid == second.event_uid
    assert first.participant_ids == ("atp:djokovic", "atp:alcaraz")
    assert second.participant_ids == ("atp:alcaraz", "atp:djokovic")


def test_event_uid_changes_with_date():
    reg = registry()
    today = normalize_event(make_raw(), reg)
    tomorrow = normalize_event(make_raw(observed_at=NOW + timedelta(days=1)), reg)
    assert today.event_uid != tomorrow.event_uid


def test_event_uid_uses_normalized_league():
    reg = registry()
    assert (
        normalize_event(make_raw(league="ATP"), reg).event_uid
        == normalize_event(make_raw(league="  atp "), reg).event_uid
    )


def test_event_uid_uses_normalized_sport():
    reg = registry()
    assert (
        normalize_event(make_raw(sport="Tennis"), reg).event_uid
        == normalize_event(make_raw(sport="tennis"), reg).event_uid
    )


def test_event_uid_differs_for_different_participants():
    reg = registry()
    other = make_raw(participants=("Djokovic N.", "Ivanov"))
    assert (
        normalize_event(make_raw(), reg).event_uid
        != normalize_event(other, reg).event_uid
    )


def test_event_uid_differs_for_different_league():
    reg = registry()
    atp = normalize_event(make_raw(league="ATP"), reg)
    atp500 = normalize_event(make_raw(league="ATP 500"), reg)
    assert atp.event_uid != atp500.event_uid


# --- name normalization ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Djokovic N.", "djokovic n"),
        ("  O'Connell   C. ", "o connell c"),
        ("Джокович Н.", "джокович н"),
        ("Team_2", "team 2"),
        ("", ""),
    ],
)
def test_normalize_name_strips_punctuation_and_case(raw, expected):
    assert normalize_name(raw) == expected


# --- registry -------------------------------------------------------------


def test_registry_resolves_unique_alias():
    reg = registry()
    assert reg.resolve("Djokovic N.") == ("atp:djokovic",)
    assert reg.resolve("джокович н.") == ("atp:djokovic",)


def test_registry_resolve_ambiguous_alias_returns_all_candidates():
    assert registry().resolve("Ivanov") == ("atp:ivanov_a", "atp:ivanov_b")


def test_registry_resolve_unknown_alias_returns_empty_tuple():
    assert registry().resolve("Nobody") == ()


def test_registry_merges_aliases_that_normalize_to_same_key(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({"A.": "atp:a", "a": "atp:b"}), encoding="utf-8")
    reg = EntityRegistry.from_json(path)
    assert reg.resolve("A") == ("atp:a", "atp:b")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"": "atp:x"},
        {"a": 1},
        {"a": []},
        {"a": ["atp:x", 2]},
        {"a": None},
    ],
)
def test_registry_rejects_invalid_fixture(tmp_path, payload):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        EntityRegistry.from_json(path)


# --- market mapping -------------------------------------------------------


@pytest.mark.parametrize("bookmaker", BOOKMAKERS)
@pytest.mark.parametrize(("market", "label"), CANONICAL_LABELS.items())
def test_canon_market_maps_every_book(bookmaker, market, label):
    assert canon_market(label, bookmaker) == market


@pytest.mark.parametrize("market", CANONICAL_LABELS)
def test_canon_market_is_idempotent_for_canonical_names(market):
    assert canon_market(market, "pari") == market


def test_canon_market_normalizes_case_and_whitespace():
    assert canon_market("  п1  ", "PARI") == "match_winner"
    assert canon_market("MATCH WINNER", "pari") == "match_winner"


@pytest.mark.parametrize(
    ("bookmaker", "label"),
    [
        ("winline", "Победа первой"),
        ("pari", "Победа участника 1"),
        ("fonbet", "Исход 1"),
        ("betboom", "Исход 1"),
    ],
)
def test_canon_market_supports_book_specific_labels(bookmaker, label):
    assert canon_market(label, bookmaker) == "match_winner"


def test_book_specific_label_is_not_shared_with_other_books():
    assert canon_market("Победа первой", "winline") == "match_winner"
    assert canon_market("Победа первой", "pari") == "unknown_победа_первой"


def test_unknown_bookmaker_falls_back_to_shared_labels():
    assert canon_market("П1", "unknownbook") == "match_winner"


def test_unknown_market_returns_unknown_slug():
    assert canon_market("Тотал 3.5", "pari") == "unknown_тотал_3_5"


def test_blank_market_returns_unknown():
    assert canon_market("  ", "pari") == "unknown"


def test_generic_total_is_not_guessed():
    assert canon_market("Тотал", "pari") == "unknown_тотал"


def test_known_market_has_full_confidence():
    assert canon_market_mapping("П1", "pari") == MarketMapping("match_winner", 1.0)


def test_unknown_market_has_zero_confidence():
    mapping = canon_market_mapping("Тотал 3.5", "pari")
    assert mapping.market == "unknown_тотал_3_5"
    assert mapping.confidence == 0.0


def test_canonical_event_is_frozen():
    ev = CanonicalEvent(
        event_uid="x",
        sport="tennis",
        league="ATP",
        participant_ids=("atp:a", "atp:b"),
        start_at=NOW,
        mapping_confidence=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        ev.sport = "basketball"  # type: ignore[misc]
