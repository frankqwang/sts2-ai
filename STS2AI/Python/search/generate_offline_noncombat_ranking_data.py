#!/usr/bin/env python3
"""Canonical entrypoint for offline non-combat ranking data generation.

The underlying implementation still lives in `generate_card_ranking_data.py`
for backward compatibility. Prefer this filename in new commands, docs, and
automation because the generator now covers map/card_reward/remove-card style
non-combat ranking samples rather than only card rewards.
"""
from __future__ import annotations

from search.generate_card_ranking_data import main


if __name__ == "__main__":
    main()
