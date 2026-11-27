"""Shared adapter contract for public, unauthenticated bookmaker sources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from betdash.db.repository import AdapterHealth
from betdash.normalize.models import RawOdd


@runtime_checkable
class Adapter(Protocol):
    """One bookmaker's line source.

    ``fetch_snapshot`` returns a fresh list of quotes and never raises into the
    collector; failures are reported through ``health`` instead.
    """

    def fetch_snapshot(self) -> list[RawOdd]:
        """Return the current pre-match quotes, or an empty list on failure."""
        ...

    def health(self) -> AdapterHealth:
        """Return the adapter's last known state."""
        ...
