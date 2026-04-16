#!/usr/bin/env python3
"""
Scrape Skada Analytics API and populate SQLite database.

Usage:
    python STS2AI/Python/skada/scrape_skada.py                # scrape everything
    python STS2AI/Python/skada/scrape_skada.py --cards-only   # just cards
    python STS2AI/Python/skada/scrape_skada.py --skip-runs    # skip the big runs table
    python STS2AI/Python/skada/scrape_skada.py --max-run-pages 10
    python STS2AI/Python/skada/scrape_skada.py --details-only --run-detail-limit 50
    python STS2AI/Python/skada/scrape_skada.py --skip-run-details

API base: http://124.223.63.165/api/
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import random
import time
import threading
import urllib.error
import sys
import os
import requests

sys.path.insert(0, os.path.dirname(__file__))
from skada_db import (
    DB_PATH,
    get_connection, create_tables,
    upsert_cards, upsert_card_floor_value, upsert_card_companions,
    upsert_card_upgrade_value, upsert_relics, upsert_encounters,
    upsert_boss_guide, upsert_runs, upsert_campfire_decisions,
    upsert_deck_size_curves, upsert_overview, update_scrape_meta,
    upsert_run_detail, record_run_detail_error,
    get_pending_run_detail_ids, get_run_detail_progress,
)

BASE_URL = "http://124.223.63.165/api"
PAGE_SIZE = 100
REQUEST_DELAY = 0.3  # seconds between requests to be polite
DETAIL_REQUEST_DELAY = 2.5  # slower cadence for per-run detail scraping
DETAIL_REQUEST_JITTER = 0.75

_THREAD_LOCAL = threading.local()


def _get_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "STS2-AI-Research/1.0",
            "Accept": "application/json",
        })
        _THREAD_LOCAL.session = session
    return session


def polite_sleep(base_delay, jitter=0.0):
    delay = max(0.0, float(base_delay)) + (random.random() * max(0.0, float(jitter)))
    if delay > 0:
        time.sleep(delay)


def fetch_json(url, retries=3):
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            resp = _get_session().get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            if attempt < retries - 1:
                response = e.response
                retry_after = response.headers.get("Retry-After") if response is not None else None
                wait = max(float(retry_after), 2 ** attempt) if retry_after else 2 ** attempt
                code = response.status_code if response is not None else "?"
                reason = response.reason if response is not None else str(e)
                print(f"  Retry {attempt+1}/{retries} after {wait:.1f}s: HTTP {code} {reason}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {url}")
                raise
        except (requests.RequestException, TimeoutError) as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts: {url}")
                raise


def fetch_paginated(endpoint, list_key, page_size=PAGE_SIZE, max_pages=None):
    """Fetch all pages from a paginated endpoint."""
    all_items = []
    page = 1
    while True:
        url = f"{BASE_URL}/{endpoint}?page={page}&page_size={page_size}"
        data = fetch_json(url)
        items = data.get(list_key, [])
        all_items.extend(items)
        pagination = data.get("pagination", {})
        total_pages = pagination.get("total_pages", 1)
        total = pagination.get("total", len(items))
        print(f"  [{endpoint}] page {page}/{total_pages} — {len(all_items)}/{total}")
        if page >= total_pages:
            break
        if max_pages and page >= max_pages:
            print(f"  [{endpoint}] stopped at max_pages={max_pages}")
            break
        page += 1
        time.sleep(REQUEST_DELAY)
    return all_items


def scrape_cards(conn):
    print("\n=== Scraping cards ===")
    cards = fetch_paginated("cards", "cards")
    upsert_cards(conn, cards)
    update_scrape_meta(conn, "cards", len(cards))
    print(f"  Saved {len(cards)} cards")
    return len(cards)


def scrape_card_floor_value(conn):
    print("\n=== Scraping card floor value ===")
    data = fetch_json(f"{BASE_URL}/deck-doctor/floor-value")
    items = data.get("items", [])
    upsert_card_floor_value(conn, items)
    update_scrape_meta(conn, "card_floor_value", len(items))
    print(f"  Saved {len(items)} card floor value entries")
    return len(items)


def scrape_card_companions(conn):
    print("\n=== Scraping card companions ===")
    data = fetch_json(f"{BASE_URL}/deck-doctor/card-companions")
    pairs = data.get("pairs", [])
    upsert_card_companions(conn, pairs)
    update_scrape_meta(conn, "card_companions", len(pairs))
    print(f"  Saved {len(pairs)} card companion pairs")
    return len(pairs)


def scrape_card_upgrade_value(conn):
    print("\n=== Scraping card upgrade value ===")
    data = fetch_json(f"{BASE_URL}/cards/upgrade-value")
    rows = data.get("rows", [])
    upsert_card_upgrade_value(conn, rows)
    update_scrape_meta(conn, "card_upgrade_value", len(rows))
    print(f"  Saved {len(rows)} upgrade value entries")
    return len(rows)


def scrape_relics(conn):
    print("\n=== Scraping relics ===")
    relics = fetch_paginated("relics", "relics")
    upsert_relics(conn, relics)
    update_scrape_meta(conn, "relics", len(relics))
    print(f"  Saved {len(relics)} relics")
    return len(relics)


def scrape_encounters(conn):
    print("\n=== Scraping encounters ===")
    encounters = fetch_paginated("encounters", "encounters")
    upsert_encounters(conn, encounters)
    update_scrape_meta(conn, "encounters", len(encounters))
    print(f"  Saved {len(encounters)} encounters")
    return len(encounters)


def scrape_boss_guide(conn):
    print("\n=== Scraping boss guide ===")
    data = fetch_json(f"{BASE_URL}/battles/boss-guide")
    bosses = data.get("bosses", [])
    upsert_boss_guide(conn, bosses)
    update_scrape_meta(conn, "boss_guide", len(bosses))
    print(f"  Saved {len(bosses)} boss entries")
    return len(bosses)


def scrape_decisions(conn):
    print("\n=== Scraping decisions ===")
    data = fetch_json(f"{BASE_URL}/battles/decisions")
    campfire = data.get("campfire", [])
    upsert_campfire_decisions(conn, campfire)
    update_scrape_meta(conn, "decisions", len(campfire))
    print(f"  Saved {len(campfire)} campfire decisions")
    return len(campfire)


def scrape_deck_size(conn):
    print("\n=== Scraping deck size curves ===")
    data = fetch_json(f"{BASE_URL}/deck-doctor/deck-size")
    characters = data.get("characters", [])
    upsert_deck_size_curves(conn, characters)
    total_points = sum(len(ch.get("curve", [])) for ch in characters)
    update_scrape_meta(conn, "deck_size", total_points)
    print(f"  Saved {len(characters)} characters, {total_points} data points")
    return total_points


def scrape_overview(conn):
    print("\n=== Scraping overview ===")
    data = fetch_json(f"{BASE_URL}/overview")
    upsert_overview(conn, data)
    update_scrape_meta(conn, "overview", 1)
    print(f"  Saved overview (total_runs={data.get('total_runs')})")
    return 1


def scrape_runs(conn, max_pages=None):
    print("\n=== Scraping runs ===")
    runs = fetch_paginated("runs", "runs", max_pages=max_pages)
    upsert_runs(conn, runs)
    update_scrape_meta(conn, "runs", len(runs))
    print(f"  Saved {len(runs)} runs")
    return len(runs)


def scrape_run_details(conn, limit=None, delay=DETAIL_REQUEST_DELAY, jitter=DETAIL_REQUEST_JITTER,
                       retry_errors=False, parallelism=1):
    print("\n=== Scraping run details ===")
    queue = get_pending_run_detail_ids(conn, limit=limit, retry_errors=retry_errors)
    if not queue:
        progress = get_run_detail_progress(conn)
        print(
            "  No pending run details "
            f"(ok={progress['ok']}, error={progress['error']}, pending={progress['pending']})"
        )
        update_scrape_meta(conn, "run_details", progress["ok"])
        return 0

    success = 0
    errors = 0
    total = len(queue)
    completed = 0

    def _fetch_detail(run_id):
        return run_id, fetch_json(f"{BASE_URL}/runs/{run_id}")

    max_workers = max(1, int(parallelism))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending_futures = {}
        queue_index = 0
        while queue_index < total or pending_futures:
            while queue_index < total and len(pending_futures) < max_workers:
                run_id = queue[queue_index]
                if queue_index > 0:
                    polite_sleep(delay, jitter)
                future = pool.submit(_fetch_detail, run_id)
                pending_futures[future] = run_id
                queue_index += 1

            done, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                run_id = pending_futures.pop(future)
                completed += 1
                try:
                    _rid, payload = future.result()
                    upsert_run_detail(conn, payload)
                    success += 1
                    progress = get_run_detail_progress(conn)
                    print(
                        f"  [run_details] {completed}/{total} run_id={run_id} ok "
                        f"(ok={progress['ok']}, error={progress['error']}, pending={progress['pending']})"
                    )
                except Exception as e:
                    errors += 1
                    record_run_detail_error(conn, run_id, e)
                    print(f"  [run_details] {completed}/{total} run_id={run_id} ERROR: {e}")
                    if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429:
                        backoff = max(30.0, delay * 8.0)
                        print(f"  [run_details] hit HTTP 429, backing off for {backoff:.1f}s")
                        time.sleep(backoff)

    progress = get_run_detail_progress(conn)
    update_scrape_meta(conn, "run_details", progress["ok"])
    print(
        f"  Saved {success} run details, recorded {errors} errors "
        f"(ok={progress['ok']}, error={progress['error']}, pending={progress['pending']})"
    )
    return success


def main():
    parser = argparse.ArgumentParser(description="Scrape Skada Analytics")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--cards-only", action="store_true", help="Only scrape cards")
    parser.add_argument("--details-only", action="store_true",
                        help="Only scrape per-run details for run_ids already stored in the DB")
    parser.add_argument("--skip-runs", action="store_true", help="Skip runs (slow)")
    parser.add_argument("--skip-run-details", action="store_true",
                        help="Skip per-run detail payloads (very slow; uses /api/runs/{id})")
    parser.add_argument("--max-run-pages", type=int, default=None,
                        help="Max pages of runs to fetch (100 runs/page)")
    parser.add_argument("--run-detail-limit", type=int, default=None,
                        help="Max number of run details to fetch this invocation")
    parser.add_argument("--run-detail-delay", type=float, default=DETAIL_REQUEST_DELAY,
                        help="Base delay between run detail requests in seconds")
    parser.add_argument("--run-detail-jitter", type=float, default=DETAIL_REQUEST_JITTER,
                        help="Random extra delay added between run detail requests in seconds")
    parser.add_argument("--run-detail-parallelism", type=int, default=1,
                        help="Max in-flight detail requests. Keep small to stay polite.")
    parser.add_argument("--retry-detail-errors", action="store_true",
                        help="Retry run details that previously failed")
    args = parser.parse_args()

    conn = get_connection(args.db)
    create_tables(conn)

    t0 = time.time()
    totals = {}

    if args.cards_only and args.details_only:
        parser.error("--cards-only and --details-only cannot be used together.")

    if args.cards_only:
        totals["cards"] = scrape_cards(conn)
    elif args.details_only:
        totals["run_details"] = scrape_run_details(
            conn,
            limit=args.run_detail_limit,
            delay=args.run_detail_delay,
            jitter=args.run_detail_jitter,
            retry_errors=args.retry_detail_errors,
            parallelism=args.run_detail_parallelism,
        )
    else:
        # Fast endpoints first
        totals["cards"] = scrape_cards(conn)
        time.sleep(REQUEST_DELAY)
        totals["card_floor_value"] = scrape_card_floor_value(conn)
        time.sleep(REQUEST_DELAY)
        totals["card_companions"] = scrape_card_companions(conn)
        time.sleep(REQUEST_DELAY)
        totals["card_upgrade_value"] = scrape_card_upgrade_value(conn)
        time.sleep(REQUEST_DELAY)
        totals["relics"] = scrape_relics(conn)
        time.sleep(REQUEST_DELAY)
        totals["encounters"] = scrape_encounters(conn)
        time.sleep(REQUEST_DELAY)
        totals["boss_guide"] = scrape_boss_guide(conn)
        time.sleep(REQUEST_DELAY)
        totals["decisions"] = scrape_decisions(conn)
        time.sleep(REQUEST_DELAY)
        totals["deck_size"] = scrape_deck_size(conn)
        time.sleep(REQUEST_DELAY)
        totals["overview"] = scrape_overview(conn)

        if not args.skip_runs:
            time.sleep(REQUEST_DELAY)
            totals["runs"] = scrape_runs(conn, max_pages=args.max_run_pages)

        if not args.skip_run_details:
            polite_sleep(args.run_detail_delay, min(args.run_detail_jitter, 0.5))
            totals["run_details"] = scrape_run_details(
                conn,
                limit=args.run_detail_limit,
                delay=args.run_detail_delay,
                jitter=args.run_detail_jitter,
                retry_errors=args.retry_detail_errors,
                parallelism=args.run_detail_parallelism,
            )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Scrape complete in {elapsed:.1f}s")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    db_path = args.db or str(DB_PATH)
    db_size = os.path.getsize(db_path) / 1024
    print(f"  Database: {db_path} ({db_size:.0f} KB)")
    conn.close()


if __name__ == "__main__":
    main()
