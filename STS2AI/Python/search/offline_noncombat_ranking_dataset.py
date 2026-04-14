#!/usr/bin/env python3
"""Canonical entrypoint for offline non-combat ranking datasets.

This module exposes the accurate public names while keeping the legacy
`matchup_dataset.py` import path working for older experiments and tests.
"""
from __future__ import annotations

from matchup_dataset import (
    MAX_OPTIONS,
    MatchupRankingDataset,
    OfflineNoncombatRankingDataset,
)

__all__ = [
    "MAX_OPTIONS",
    "OfflineNoncombatRankingDataset",
    "MatchupRankingDataset",
]
