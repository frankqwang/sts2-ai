"""4-way combat-by-combat comparison across baseline / w02 / v4 / v6.

Variant of compare_trajectory_combats.py that aligns 4 policies.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def _lower(x: Any) -> str:
    return str(x or '').strip().lower()


def _raw(rec): return rec.get('raw_state') if isinstance(rec.get('raw_state'), dict) else {}
def _next(rec): return rec.get('next_state') if isinstance(rec.get('next_state'), dict) else {}
def _player(state): return state.get('player') if isinstance(state.get('player'), dict) else {}


def _enemy_sig(state):
    enemies = state.get('enemies') or state.get('monsters') or []
    if not isinstance(enemies, list):
        return []
    out = []
    for e in enemies:
        if isinstance(e, dict):
            n = e.get('name') or e.get('monster_id') or '?'
            out.append(str(n))
    return out


def _summarize_combats(records):
    combats = []
    cur = None
    for rec in records:
        state = _raw(rec)
        st = _lower(state.get('state_type'))
        floor = int(rec.get('floor') or 0)
        if st in {'combat', 'monster', 'elite', 'boss'}:
            if cur is None or cur['floor'] != floor or cur['room'] != st:
                if cur is not None:
                    combats.append(cur)
                pl = _player(state)
                cur = {
                    'floor': floor, 'room': st,
                    'start_hp': pl.get('current_hp', pl.get('hp')),
                    'enemies': _enemy_sig(state),
                    'turns': 0, 'cards': [], 'potions': [],
                    'last_state': state,
                }
            cur['turns'] = max(cur['turns'], int(rec.get('step_index') or 0))
            action = rec.get('chosen_action') or {}
            atype = _lower(action.get('action') or action.get('type'))
            label = action.get('label') or action.get('action') or atype
            if atype == 'end_turn':
                cur['turns'] += 0  # handled by step_index
            if atype == 'play_card':
                cur['cards'].append(label)
            elif atype == 'use_potion':
                cur['potions'].append(label)
            cur['last_state'] = _next(rec) or state
        else:
            if cur is not None:
                combats.append(cur)
                cur = None
    if cur is not None:
        combats.append(cur)
    # finalize
    out = []
    for c in combats:
        last = c['last_state']
        lp = _player(last)
        final_hp = lp.get('current_hp', lp.get('hp'))
        outcome = 'ongoing'
        final_st = _lower(last.get('state_type'))
        if final_st == 'game_over' or last.get('terminal'):
            outcome = 'death'
        elif final_st not in {'combat', 'monster', 'elite', 'boss'}:
            outcome = 'win'
        dmg = 0
        if c['start_hp'] is not None and final_hp is not None:
            try: dmg = int(c['start_hp']) - int(final_hp)
            except: pass
        out.append({
            'floor': c['floor'], 'room': c['room'], 'enemies': c['enemies'],
            'start_hp': c['start_hp'], 'final_hp': final_hp, 'dmg_in': dmg,
            'n_cards': len(c['cards']),
            'outcome': outcome,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-dir', required=True)
    p.add_argument('--w02-dir', required=True)
    p.add_argument('--v4-dir', required=True)
    p.add_argument('--v6-dir', required=True)
    p.add_argument('--seeds', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    dirs = {'base': Path(args.baseline_dir), 'w02': Path(args.w02_dir), 'v4': Path(args.v4_dir), 'v6': Path(args.v6_dir)}
    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    lines = []

    for seed in seeds:
        lines.append(f'\n==================== {seed} ====================')
        seed_data = {}
        for label, d in dirs.items():
            jp = d / f'{seed}_trajectory.jsonl'
            if not jp.exists():
                lines.append(f'  [{label}] missing')
                continue
            recs = _load(jp)
            combats = _summarize_combats(recs)
            final = _next(recs[-1]) if recs else {}
            pl = _player(final)
            seed_data[label] = {
                'combats': combats,
                'max_floor': max((int(r.get('floor') or 0) for r in recs), default=0),
                'final_hp': pl.get('current_hp', pl.get('hp')),
                'nrec': len(recs),
            }
        # Summary row
        lines.append(f"{'pol':<5}{'records':>10}{'max_flr':>10}{'final_hp':>10}{'combats':>10}")
        for lbl in ('base', 'w02', 'v4', 'v6'):
            d = seed_data.get(lbl, {})
            if not d: continue
            lines.append(f"{lbl:<5}{d['nrec']:>10}{d['max_floor']:>10}{str(d['final_hp']):>10}{len(d['combats']):>10}")

        # Align by floor/room
        pool = defaultdict(dict)
        for lbl in ('base', 'w02', 'v4', 'v6'):
            for c in seed_data.get(lbl, {}).get('combats', []):
                pool[(c['floor'], c['room'])][lbl] = c
        lines.append('\nPer combat:')
        for (floor, room), m in sorted(pool.items()):
            enemies = ''
            for lbl in ('base','w02','v4','v6'):
                if lbl in m:
                    enemies = ','.join(m[lbl]['enemies'])
                    break
            lines.append(f'\n  floor {floor} [{room}] vs {enemies}:')
            lines.append(f"    {'pol':<5}{'dmg_in':>8}{'start':>8}{'end':>8}{'cards':>7}{'outcome':>10}")
            for lbl in ('base', 'w02', 'v4', 'v6'):
                c = m.get(lbl)
                if not c:
                    lines.append(f"    {lbl:<5}  [not reached]")
                    continue
                lines.append(f"    {lbl:<5}{c['dmg_in']:>8}{str(c['start_hp']):>8}{str(c['final_hp']):>8}{c['n_cards']:>7}{c['outcome']:>10}")

    Path(args.output).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
