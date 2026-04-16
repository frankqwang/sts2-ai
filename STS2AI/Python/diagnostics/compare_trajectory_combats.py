"""轨迹战斗对比：对比不同轨迹的战斗细节。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_trajectory(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def _lower(x: Any) -> str:
    return str(x or '').strip().lower()


def _raw_state(rec: dict) -> dict:
    return rec.get('raw_state') if isinstance(rec.get('raw_state'), dict) else {}


def _next_state(rec: dict) -> dict:
    return rec.get('next_state') if isinstance(rec.get('next_state'), dict) else {}


def _get_player(state: dict) -> dict:
    return state.get('player') if isinstance(state.get('player'), dict) else {}


def _enemy_signature(state: dict) -> list[str]:
    enemies = state.get('enemies') or state.get('monsters') or []
    names: list[str] = []
    if isinstance(enemies, list):
        for e in enemies:
            if isinstance(e, dict):
                n = e.get('name') or e.get('monster_id') or '?'
                hp = e.get('hp') if e.get('hp') is not None else e.get('current_hp')
                max_hp = e.get('max_hp')
                if max_hp:
                    names.append(f"{n}[{hp}/{max_hp}]")
                else:
                    names.append(str(n))
    return names


def _summarize_combats(records: list[dict]) -> list[dict]:
    """Group records into combats (contiguous combat/monster/elite/boss segments)."""
    combats: list[dict] = []
    cur: dict | None = None
    for rec in records:
        state = _raw_state(rec)
        st = _lower(state.get('state_type'))
        floor = int(rec.get('floor') or 0)
        if st in {'combat', 'monster', 'elite', 'boss'}:
            if cur is None or cur.get('floor') != floor or _lower(cur.get('room')) != st:
                if cur is not None:
                    combats.append(cur)
                player = _get_player(state)
                cur = {
                    'floor': floor,
                    'room': st,
                    'start_hp': player.get('current_hp', player.get('hp')),
                    'max_hp': player.get('max_hp'),
                    'enemies_initial': _enemy_signature(state),
                    'turns': 0,
                    'actions': [],
                    'end_turn_count': 0,
                    'cards_played': [],
                    'potions_used': [],
                    'last_state': state,
                }
            cur['turns'] = max(cur['turns'], int(rec.get('step_index') or 0))
            action = rec.get('chosen_action') or {}
            atype = _lower(action.get('action') or action.get('type'))
            label = action.get('label') or action.get('action') or atype
            cur['actions'].append((int(rec.get('step_index') or 0), atype, label))
            if atype == 'end_turn':
                cur['end_turn_count'] += 1
            elif atype == 'play_card':
                cur['cards_played'].append(label)
            elif atype == 'use_potion':
                cur['potions_used'].append(label)
            cur['last_state'] = _next_state(rec) or state
        else:
            if cur is not None:
                combats.append(cur)
                cur = None
    if cur is not None:
        combats.append(cur)

    # Post-process: compute outcome, final HP
    summaries = []
    for c in combats:
        last = c.get('last_state') or {}
        last_player = _get_player(last)
        final_hp = last_player.get('current_hp', last_player.get('hp'))
        # Determine outcome
        outcome = 'ongoing'
        final_st = _lower(last.get('state_type'))
        if final_st == 'game_over' or last.get('terminal'):
            outcome = 'death'
        elif final_st not in {'combat', 'monster', 'elite', 'boss'}:
            outcome = 'win'
        damage_taken = 0
        if c['start_hp'] is not None and final_hp is not None:
            try:
                damage_taken = int(c['start_hp']) - int(final_hp)
            except (TypeError, ValueError):
                pass
        summaries.append({
            'floor': c['floor'],
            'room': c['room'],
            'enemies': c['enemies_initial'],
            'turns': c['end_turn_count'],
            'start_hp': c['start_hp'],
            'final_hp': final_hp,
            'damage_taken': damage_taken,
            'n_actions': len(c['actions']),
            'cards_played': c['cards_played'],
            'potions_used': c['potions_used'],
            'outcome': outcome,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--w02-dir', required=True)
    parser.add_argument('--v4-dir', required=True)
    parser.add_argument('--seeds', required=True, help='Comma-separated seed list.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    label_dirs = {
        'base': Path(args.baseline_dir),
        'w02': Path(args.w02_dir),
        'v4': Path(args.v4_dir),
    }
    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]

    report_lines: list[str] = []
    for seed in seeds:
        report_lines.append(f'\n==================== {seed} ====================')
        seed_data = {}
        for label, d in label_dirs.items():
            p = d / f'{seed}_trajectory.jsonl'
            if not p.exists():
                report_lines.append(f'  [{label}] missing: {p}')
                continue
            recs = _load_trajectory(p)
            combats = _summarize_combats(recs)
            last_floor = max((int(r.get('floor') or 0) for r in recs), default=0)
            final_state = _next_state(recs[-1]) if recs else {}
            final_player = _get_player(final_state)
            final_hp = final_player.get('current_hp', final_player.get('hp'))
            outcome = _lower(final_state.get('state_type'))
            seed_data[label] = {
                'combats': combats,
                'last_floor': last_floor,
                'final_hp': final_hp,
                'outcome': outcome,
                'n_records': len(recs),
            }

        # Per-label summary
        report_lines.append(f"{'pol':<5}{'records':>10}{'max_floor':>11}{'final_hp':>10}{'combats':>10}")
        for label in ('base', 'w02', 'v4'):
            d = seed_data.get(label, {})
            if not d:
                continue
            report_lines.append(f"{label:<5}{d['n_records']:>10}{d['last_floor']:>11}{str(d['final_hp']):>10}{len(d['combats']):>10}")

        # Per-combat diff (aligned by floor+room)
        combat_map: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
        for label in ('base', 'w02', 'v4'):
            for c in seed_data.get(label, {}).get('combats', []):
                combat_map[(c['floor'], c['room'])][label] = c

        report_lines.append('\nCombat-by-combat:')
        for (floor, room), policies in sorted(combat_map.items()):
            enemies = ''
            for label in ('base', 'w02', 'v4'):
                c = policies.get(label)
                if c:
                    enemies = ','.join(c['enemies'])
                    break
            report_lines.append(f'\n  floor {floor} [{room}] vs {enemies}:')
            report_lines.append(f"    {'pol':<5}{'turns':>6}{'dmg_in':>8}{'start_hp':>10}{'end_hp':>8}{'outcome':>10}  cards_played")
            for label in ('base', 'w02', 'v4'):
                c = policies.get(label)
                if not c:
                    report_lines.append(f'    {label:<5}  [not reached]')
                    continue
                cards_str = ','.join(c['cards_played'][:8])
                report_lines.append(
                    f"    {label:<5}{c['turns']:>6}{c['damage_taken']:>8}"
                    f"{str(c['start_hp']):>10}{str(c['final_hp']):>8}{c['outcome']:>10}  {cards_str}"
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
