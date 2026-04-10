#!/usr/bin/env python3
"""Training pipeline smoke tests — no Godot required.

Verifies all Python training components work correctly:
  - Module imports
  - Vocab loading + encoder output shapes
  - PPO / Combat network forward passes
  - StructuredRolloutBuffer + GAE
  - PPOTrainerV2 gradient flow
  - quality_check() accept/reject logic

Usage:
    cd STS2AI/Python
    python -m pytest test_training_smoke.py -v
    python -m pytest test_training_smoke.py -v -k "test_ppo"  # subset
"""

from __future__ import annotations

import _path_init  # noqa: F401  (adds STS2AI/Python library dirs to sys.path)

import copy
import sys
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _import_or_skip(module_name: str):
    return pytest.importorskip(module_name, reason="Historical script is not part of the STS2AI mainline contract.")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def vocab():
    from vocab import load_vocab
    return load_vocab()


def _make_synthetic_state(screen_type: str = "map") -> dict:
    """Build a minimal synthetic game state dict for encoder tests."""
    return {
        "state_type": screen_type,
        "run": {
            "floor": 5,
            "act": 1,
            "ascension_level": 0,
            "next_boss_id": "CEREMONIAL_BEAST_BOSS",
            "next_boss_name": "仪式兽",
            "next_boss_archetype": "ceremonial_beast_boss",
        },
        "player": {
            "hp": 60,
            "max_hp": 80,
            "energy": 3,
            "max_energy": 3,
            "gold": 99,
            "block": 0,
            "deck": [
                {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "ATTACK",
                 "upgrades": 0, "rarity": "BASIC"},
                {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "SKILL",
                 "upgrades": 0, "rarity": "BASIC"},
            ],
            "relics": [
                {"id": "BURNING_BLOOD"},
            ],
            "potions": [],
        },
        "battle": {
            "hand": [
                {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "ATTACK",
                 "upgrades": 0, "rarity": "BASIC"},
            ],
            "enemies": [
                {"id": "JAW_WORM", "name": "Jaw Worm", "hp": 40, "max_hp": 44,
                 "block": 0, "intent": "attack", "powers": []},
            ],
            "is_play_phase": True,
        } if screen_type in ("combat", "monster", "elite", "boss") else None,
        "map": {
            "nodes": [
                {"type": "monster", "x": 0, "y": 1},
                {"type": "elite", "x": 1, "y": 1},
            ],
        } if screen_type == "map" else None,
        "legal_actions": [],
    }


def _make_synthetic_actions(screen_type: str = "map") -> list[dict]:
    """Build synthetic legal actions matching the screen type."""
    if screen_type in ("combat", "monster", "elite", "boss"):
        return [
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD",
             "card_index": 0, "target_id": "JAW_WORM"},
            {"action": "end_turn"},
        ]
    elif screen_type == "map":
        return [
            {"action": "choose_map_node", "index": 0, "node_type": "monster"},
            {"action": "choose_map_node", "index": 1, "node_type": "elite"},
        ]
    elif screen_type == "card_reward":
        return [
            {"action": "select_card_reward", "card_id": "STRIKE_IRONCLAD", "index": 0},
            {"action": "skip"},
        ]
    return [{"action": "proceed"}]


def _make_v1_combat_state(state_type: str = "boss", floor: int = 16) -> dict:
    enemy_entity = "jaw-worm-0"
    return {
        "state_type": state_type,
        "run": {
            "floor": floor,
            "act": 1,
            "ascension_level": 0,
        },
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "player": {
                "character": "IRONCLAD",
                "hp": 60,
                "max_hp": 80,
                "block": 0,
                "energy": 1,
                "max_energy": 3,
                "gold": 99,
                "draw_pile_count": 8,
                "discard_pile_count": 0,
                "exhaust_pile_count": 0,
                "potions": [],
                "hand": [
                    {
                        "index": 0,
                        "id": "STRIKE_IRONCLAD",
                        "name": "Strike",
                        "type": "ATTACK",
                        "cost": 1,
                        "target_type": "enemy",
                        "can_play": True,
                    }
                ],
            },
            "enemies": [
                {
                    "entity_id": enemy_entity,
                    "combat_id": 0,
                    "name": "Jaw Worm",
                    "hp": 18,
                    "max_hp": 18,
                    "block": 0,
                    "intents": [{"type": "attack", "label": "10"}],
                    "status": [],
                }
            ],
        },
        "player": {
            "hp": 60,
            "max_hp": 80,
            "gold": 99,
            "potions": [],
        },
        "legal_actions": [
            {"action": "play_card", "card_index": 0, "target": enemy_entity},
            {"action": "end_turn"},
        ],
    }


def _make_teacher_combat_state(hand_cards: list[dict], *, enemy_hp: int = 30, block: int = 0) -> dict:
    enemy_entity = "cultist-0"
    return {
        "state_type": "monster",
        "run": {
            "floor": 7,
            "act": 1,
        },
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "player": {
                "character": "IRONCLAD",
                "hp": 55,
                "max_hp": 80,
                "block": block,
                "energy": 3,
                "max_energy": 3,
                "gold": 99,
                "draw_pile_count": 5,
                "discard_pile_count": 0,
                "exhaust_pile_count": 0,
                "potions": [],
                "hand": hand_cards,
            },
            "enemies": [
                {
                    "entity_id": enemy_entity,
                    "combat_id": 0,
                    "name": "Cultist",
                    "hp": enemy_hp,
                    "max_hp": enemy_hp,
                    "block": 0,
                    "intents": [{"type": "attack", "label": "8", "total_damage": 8}],
                    "status": [],
                }
            ],
        },
        "player": {
            "hp": 55,
            "max_hp": 80,
            "gold": 99,
            "potions": [],
        },
    }


class _FakeCombatTurnEnv:
    def __init__(self, state: dict):
        self._current = copy.deepcopy(state)
        self._saved: dict[str, dict] = {}
        self._next_id = 0

    def save_state(self) -> str:
        state_id = f"s{self._next_id}"
        self._next_id += 1
        self._saved[state_id] = copy.deepcopy(self._current)
        return state_id

    def load_state(self, state_id: str) -> dict:
        self._current = copy.deepcopy(self._saved[state_id])
        return copy.deepcopy(self._current)

    def delete_state(self, state_id: str) -> bool:
        self._saved.pop(state_id, None)
        return True

    def clear_state_cache(self) -> bool:
        self._saved.clear()
        return True

    def act(self, payload: dict) -> dict:
        action = str(payload.get("action") or "").lower()
        if action == "end_turn":
            return copy.deepcopy(self._current)
        battle = self._current["battle"]
        player = battle["player"]
        hand = battle["player"]["hand"]
        enemies = battle["enemies"]
        enemy = enemies[0]
        if action == "use_potion":
            player["potions"] = []
        elif action == "play_card":
            card_index = int(payload.get("card_index", payload.get("index", 0)))
            card = hand.pop(card_index)
            card_id = str(card.get("id") or "")
            player["energy"] = max(0, int(player.get("energy", 0)) - int(card.get("cost", 0)))
            if card_id == "BASH":
                damage = 8
                enemy["hp"] = max(0, int(enemy["hp"]) - damage)
                enemy["status"] = [{"id": "vulnerable", "amount": 2}]
            elif card_id == "STRIKE_IRONCLAD":
                vulnerable = 1.5 if any(s.get("id") == "vulnerable" for s in enemy.get("status", [])) else 1.0
                damage = int(round(6 * vulnerable))
                enemy["hp"] = max(0, int(enemy["hp"]) - damage)
            elif card_id == "DEFEND_IRONCLAD":
                player["block"] = int(player.get("block", 0)) + 5
            elif card_id == "BODY_SLAM":
                enemy["hp"] = max(0, int(enemy["hp"]) - int(player.get("block", 0)))
        if int(enemy["hp"]) <= 0:
            self._current["state_type"] = "combat_rewards"
            self._current["legal_actions"] = [{"action": "proceed", "is_enabled": True}]
        else:
            legal_actions = []
            for idx, card in enumerate(hand):
                legal_actions.append(
                    {
                        "action": "play_card",
                        "card_index": idx,
                        "index": idx,
                        "label": card.get("name") or card.get("id"),
                        "card_id": card.get("id"),
                        "target_id": enemy["entity_id"],
                        "is_enabled": int(card.get("cost", 0)) <= int(player.get("energy", 0)),
                    }
                )
            legal_actions.append({"action": "end_turn", "is_enabled": True})
            self._current["legal_actions"] = legal_actions
        return copy.deepcopy(self._current)


class _DummyBaselinePolicy:
    def score(self, state: dict, legal_actions: list[dict]) -> dict:
        return {
            "logits": np.zeros(len(legal_actions), dtype=np.float32),
            "probs": np.ones(len(legal_actions), dtype=np.float32) / max(1, len(legal_actions)),
            "value": 0.0,
            "best_index": 0,
        }


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_vocab(self):
        from vocab import load_vocab, Vocab
        assert Vocab is not None

    def test_encoder(self):
        from rl_encoder_v2 import build_structured_state, build_structured_actions

    def test_ppo_network(self):
        from rl_policy_v2 import FullRunPolicyNetworkV2, PPOTrainerV2, StructuredRolloutBuffer

    def test_combat_network(self):
        from combat_nn import CombatPolicyValueNetwork, build_combat_features

    def test_reward_shaping(self):
        from rl_reward_shaping import shaped_reward

    def test_mcts_core(self):
        from mcts_core import MCTSConfig


# ---------------------------------------------------------------------------
# 2. Vocab tests
# ---------------------------------------------------------------------------

class TestVocab:
    def test_load(self, vocab):
        assert vocab.card_vocab_size > 10
        assert vocab.relic_vocab_size > 5
        assert vocab.potion_vocab_size > 3
        assert vocab.monster_vocab_size > 5

    def test_lookup(self, vocab):
        # Known cards
        idx = vocab.card_idx("strike_ironclad")
        assert idx >= 2  # 0=pad, 1=unk
        # Unknown → unk
        assert vocab.card_idx("nonexistent_card_xyz") == 1

    def test_special_tokens(self, vocab):
        assert vocab.card_idx("") == 1  # unk


# ---------------------------------------------------------------------------
# 3. Encoder tests
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_structured_state_shape(self, vocab):
        from rl_encoder_v2 import (
            build_structured_state, SCALAR_DIM, MAX_DECK_SIZE,
            MAX_HAND_SIZE, MAX_RELICS, MAX_POTIONS, MAX_ENEMIES,
            CARD_AUX_DIM,
        )

        state = _make_synthetic_state("map")
        ss = build_structured_state(state, vocab)

        assert ss.scalars.shape == (SCALAR_DIM,)
        assert ss.deck_ids.shape == (MAX_DECK_SIZE,)
        assert ss.deck_aux.shape == (MAX_DECK_SIZE, CARD_AUX_DIM)
        assert ss.deck_mask.shape == (MAX_DECK_SIZE,)
        assert ss.deck_mask.sum() == 2  # 2 cards in synthetic deck
        assert ss.relic_ids.shape == (MAX_RELICS,)
        assert ss.relic_mask.sum() == 1  # 1 relic
        assert ss.hand_ids.shape == (MAX_HAND_SIZE,)
        assert ss.enemy_ids.shape == (MAX_ENEMIES,)

    def test_structured_actions_shape(self, vocab):
        from rl_encoder_v2 import build_structured_actions, MAX_ACTIONS

        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        sa = build_structured_actions(state, actions, vocab)

        assert sa.action_type_ids.shape == (MAX_ACTIONS,)
        assert sa.action_mask.shape == (MAX_ACTIONS,)
        assert sa.action_mask.sum() == 2  # 2 map node actions
        assert sa.num_actions == 2

    def test_combat_features_shape(self, vocab):
        from combat_nn import (
            build_combat_features, build_combat_action_features,
            COMBAT_SCALAR_DIM, MAX_ACTIONS,
        )
        from rl_encoder_v2 import MAX_HAND_SIZE, MAX_ENEMIES, CARD_AUX_DIM, ENEMY_AUX_DIM
        from combat_nn import COMBAT_EXTRA_SCALAR_DIM

        state = _make_synthetic_state("combat")
        sf = build_combat_features(state, vocab)

        assert sf["scalars"].shape == (COMBAT_SCALAR_DIM,)
        assert sf["extra_scalars"].shape == (COMBAT_EXTRA_SCALAR_DIM,), \
            f"extra_scalars shape mismatch: {sf['extra_scalars'].shape}"
        assert sf["hand_ids"].shape == (MAX_HAND_SIZE,)
        assert sf["hand_aux"].shape == (MAX_HAND_SIZE, CARD_AUX_DIM)
        assert sf["hand_mask"].dtype == bool
        assert sf["enemy_ids"].shape == (MAX_ENEMIES,)
        assert sf["enemy_aux"].shape == (MAX_ENEMIES, ENEMY_AUX_DIM), \
            f"enemy_aux shape mismatch: got {sf['enemy_aux'].shape}, expected {(MAX_ENEMIES, ENEMY_AUX_DIM)}"

        actions = _make_synthetic_actions("combat")
        af = build_combat_action_features(state, actions, vocab)
        assert af["action_type_ids"].shape == (MAX_ACTIONS,)
        assert af["action_mask"].shape == (MAX_ACTIONS,)
        assert af["action_mask"].sum() == 2


# ---------------------------------------------------------------------------
# 4. Network forward pass tests
# ---------------------------------------------------------------------------

class TestPPONetwork:
    def test_forward(self, vocab):
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from rl_encoder_v2 import build_structured_state, build_structured_actions, MAX_ACTIONS
        from rl_policy_v2 import _structured_state_to_numpy_dict, _structured_actions_to_numpy_dict

        net = FullRunPolicyNetworkV2(vocab=vocab)
        net.eval()
        assert net.param_count() > 0

        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, actions, vocab)

        # Build tensors
        state_t = {}
        for k, v in _structured_state_to_numpy_dict(ss).items():
            t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
            if "ids" in k or "idx" in k or "types" in k or "count" in k:
                state_t[k] = t.long()
            elif "mask" in k:
                state_t[k] = t.bool()
            else:
                state_t[k] = t.float()

        action_t = {}
        for k, v in _structured_actions_to_numpy_dict(sa).items():
            t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
            if "ids" in k or "types" in k or "indices" in k:
                action_t[k] = t.long()
            elif "mask" in k:
                action_t[k] = t.bool()
            else:
                action_t[k] = t.float()

        with torch.no_grad():
            logits, values, deck_q, boss_ready, action_adv = net.forward(state_t, action_t)

        assert logits.shape == (1, MAX_ACTIONS)
        assert values.shape == (1,)
        assert deck_q.shape == (1,)
        assert boss_ready.shape == (1,)
        assert action_adv.shape == (1, MAX_ACTIONS)
        assert torch.isfinite(logits[0, :sa.num_actions]).all()
        assert torch.isfinite(values).all()
        assert torch.isfinite(boss_ready).all()
        assert torch.isfinite(action_adv).all()

    def test_get_action_and_value(self, vocab):
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from rl_encoder_v2 import build_structured_state, build_structured_actions
        from rl_policy_v2 import _structured_state_to_numpy_dict, _structured_actions_to_numpy_dict

        net = FullRunPolicyNetworkV2(vocab=vocab)
        net.eval()

        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, actions, vocab)

        state_t = {}
        for k, v in _structured_state_to_numpy_dict(ss).items():
            t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
            if "ids" in k or "idx" in k or "types" in k or "count" in k:
                state_t[k] = t.long()
            elif "mask" in k:
                state_t[k] = t.bool()
            else:
                state_t[k] = t.float()
        action_t = {}
        for k, v in _structured_actions_to_numpy_dict(sa).items():
            t = torch.tensor(v).unsqueeze(0) if isinstance(v, np.ndarray) else torch.tensor([v])
            if "ids" in k or "types" in k or "indices" in k:
                action_t[k] = t.long()
            elif "mask" in k:
                action_t[k] = t.bool()
            else:
                action_t[k] = t.float()

        with torch.no_grad():
            act_idx, log_prob, entropy, value = net.get_action_and_value(state_t, action_t)

        assert act_idx.shape == (1,)
        assert 0 <= act_idx.item() < sa.num_actions
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(value).all()


class TestCombatNetwork:
    def test_forward(self, vocab):
        from combat_nn import CombatPolicyValueNetwork, build_combat_features, build_combat_action_features, MAX_ACTIONS

        net = CombatPolicyValueNetwork(vocab=vocab)
        net.eval()
        assert net.param_count() > 0

        state = _make_synthetic_state("combat")
        actions = _make_synthetic_actions("combat")
        sf = build_combat_features(state, vocab)
        af = build_combat_action_features(state, actions, vocab)

        # Batch dimension
        state_t = {}
        for k, v in sf.items():
            t = torch.tensor(v).unsqueeze(0)
            if v.dtype in (np.int64, np.int32):
                state_t[k] = t.long()
            elif v.dtype == bool:
                state_t[k] = t.bool()
            else:
                state_t[k] = t.float()

        action_t = {}
        for k, v in af.items():
            t = torch.tensor(v).unsqueeze(0)
            if v.dtype in (np.int64, np.int32):
                action_t[k] = t.long()
            elif v.dtype == bool:
                action_t[k] = t.bool()
            else:
                action_t[k] = t.float()

        with torch.no_grad():
            logits, value = net.forward(state_t, action_t)

        assert logits.shape == (1, MAX_ACTIONS)
        assert value.shape == (1,)
        assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# 5. Buffer + GAE tests
# ---------------------------------------------------------------------------

class TestBuffer:
    def test_add_and_len(self, vocab):
        from rl_policy_v2 import StructuredRolloutBuffer
        from rl_encoder_v2 import build_structured_state, build_structured_actions

        buf = StructuredRolloutBuffer()
        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, actions, vocab)

        for i in range(10):
            buf.add(ss, sa, action_idx=0, log_prob=-0.5, reward=0.1, value=0.5, done=(i == 9))

        assert len(buf) == 10

    def test_gae(self, vocab):
        from rl_policy_v2 import StructuredRolloutBuffer
        from rl_encoder_v2 import build_structured_state, build_structured_actions

        buf = StructuredRolloutBuffer()
        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, actions, vocab)

        for i in range(10):
            buf.add(ss, sa, action_idx=0, log_prob=-0.5, reward=0.1 * (i + 1), value=0.5, done=(i == 9))
        buf.set_floor_targets(5.0)
        buf.compute_gae()

        assert len(buf.advantages) == 10
        assert len(buf.returns) == 10
        assert all(np.isfinite(a) for a in buf.advantages)
        assert all(np.isfinite(r) for r in buf.returns)

    def test_clear(self, vocab):
        from rl_policy_v2 import StructuredRolloutBuffer
        from rl_encoder_v2 import build_structured_state, build_structured_actions

        buf = StructuredRolloutBuffer()
        ss = build_structured_state(_make_synthetic_state("map"), vocab)
        sa = build_structured_actions(_make_synthetic_state("map"), _make_synthetic_actions("map"), vocab)
        buf.add(ss, sa, 0, -0.5, 0.1, 0.5, False)
        buf.clear()
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# 6. PPO update test
# ---------------------------------------------------------------------------

class TestPPOUpdate:
    def test_gradient_flow(self, vocab):
        from rl_policy_v2 import FullRunPolicyNetworkV2, PPOTrainerV2, StructuredRolloutBuffer
        from rl_encoder_v2 import build_structured_state, build_structured_actions

        net = FullRunPolicyNetworkV2(vocab=vocab)
        trainer = PPOTrainerV2(network=net, lr=1e-3, ppo_epochs=1, minibatch_size=8)

        buf = StructuredRolloutBuffer()
        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, actions, vocab)

        for i in range(20):
            buf.add(ss, sa, action_idx=i % 2, log_prob=-0.5, reward=0.1, value=0.5, done=(i == 19))
        buf.set_floor_targets(5.0)
        buf.compute_gae()

        metrics = trainer.update(buf)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["entropy"])

        # Verify at least one param got gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad, "No gradients flowed through the network"


# ---------------------------------------------------------------------------
# 7. Quality gate test
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_pass(self):
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            steps: int
            max_floor: int
            outcome: str
            elapsed_s: float
            total_reward: float

        from training_health import quality_check

        results = [
            FakeResult(steps=100, max_floor=5, outcome="death", elapsed_s=10, total_reward=-0.5),
            FakeResult(steps=150, max_floor=3, outcome="death", elapsed_s=12, total_reward=-0.3),
        ]
        passed, reason = quality_check(results, iteration=0)
        assert passed, f"Should pass but got: {reason}"

    def test_reject_low_steps(self):
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            steps: int
            max_floor: int
            outcome: str
            elapsed_s: float
            total_reward: float

        from training_health import quality_check

        results = [
            FakeResult(steps=3, max_floor=1, outcome="death", elapsed_s=1, total_reward=-1.0),
            FakeResult(steps=5, max_floor=1, outcome="truncated", elapsed_s=2, total_reward=-1.0),
        ]
        passed, reason = quality_check(results, iteration=0)
        assert not passed

    def test_reject_nan(self):
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            steps: int
            max_floor: int
            outcome: str
            elapsed_s: float
            total_reward: float

        from training_health import quality_check

        results = [
            FakeResult(steps=100, max_floor=5, outcome="death", elapsed_s=10, total_reward=float("nan")),
        ]
        passed, reason = quality_check(results, iteration=0)
        assert not passed
        assert "nan" in reason.lower()


# ---------------------------------------------------------------------------
# 8. Reward shaping test
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_shaped_reward_basic(self):
        from rl_reward_shaping import shaped_reward

        prev = {
            "run": {"floor": 1, "act": 1},
            "player": {"hp": 70, "max_hp": 80, "gold": 50, "relics": []},
        }
        curr = {
            "run": {"floor": 2, "act": 1},
            "player": {"hp": 65, "max_hp": 80, "gold": 60,
                        "relics": [{"id": "BURNING_BLOOD"}]},
        }
        r = shaped_reward(prev, curr, raw_terminal_reward=0.0, done=False)
        assert np.isfinite(r)
        # Floor increased → should be positive reward
        assert r > 0, f"Floor increase should give positive reward, got {r}"

    def test_terminal_reward(self):
        from rl_reward_shaping import shaped_reward

        prev = {
            "run": {"floor": 10, "act": 2},
            "player": {"hp": 50, "max_hp": 80, "gold": 100,
                        "relics": [{"id": "BURNING_BLOOD"}]},
        }
        curr = {
            "state_type": "game_over",
            "run": {"floor": 10, "act": 2},
            "player": {"hp": 0, "max_hp": 80, "gold": 100,
                        "relics": [{"id": "BURNING_BLOOD"}]},
        }
        r = shaped_reward(prev, curr, raw_terminal_reward=-1.0, done=True)
        assert np.isfinite(r)
        # New design: death at Act 2 floor 10 (total_floor=27) gets floor bonus
        # terminal = -1.0 + min(27/17, 1.0) = 0.0 — dying far is almost neutral
        assert r <= 0, "Death at Act 2 floor 10 should not be positive"

        # Early death (floor 2) should be clearly negative
        early_prev = {"run": {"floor": 2, "act": 1},
                      "player": {"hp": 30, "max_hp": 80, "gold": 50,
                                  "relics": [], "deck": [], "potions": []}}
        early_curr = {"state_type": "game_over", "run": {"floor": 2, "act": 1},
                      "player": {"hp": 0, "max_hp": 80, "gold": 0,
                                  "relics": [], "deck": [], "potions": []}}
        r_early = shaped_reward(early_prev, early_curr, raw_terminal_reward=-1.0, done=True)
        assert r_early < -0.5, f"Early death (floor 2) should be very negative, got {r_early}"

    def test_screen_local_delta_reward_card_reward_positive(self):
        from rl_reward_shaping import screen_local_delta_reward

        prev = {
            "state_type": "card_reward",
            "run": {"floor": 8, "act": 1},
            "player": {
                "hp": 55, "max_hp": 80, "gold": 60, "relics": [],
                "deck": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "cost": 1, "upgrades": 0}] * 8
                        + [{"id": "DEFEND_IRONCLAD", "type": "SKILL", "cost": 1, "upgrades": 0}] * 4,
                "potions": [],
            },
        }
        curr = {
            "state_type": "map",
            "run": {"floor": 8, "act": 1},
            "player": {
                "hp": 55, "max_hp": 80, "gold": 60, "relics": [],
                "deck": prev["player"]["deck"] + [
                    {"id": "POMMEL_STRIKE", "type": "ATTACK", "cost": 1, "upgrades": 0},
                ],
                "potions": [],
            },
        }

        r = screen_local_delta_reward(prev, curr, "card_reward")
        assert 0.0 < r <= 0.05

    def test_screen_local_delta_reward_ignores_event(self):
        from rl_reward_shaping import screen_local_delta_reward

        prev = _make_synthetic_state("event")
        curr = _make_synthetic_state("event")
        curr["player"]["gold"] = 200

        assert screen_local_delta_reward(prev, curr, "event") == 0.0

    def test_screen_local_delta_reward_includes_map(self):
        from rl_reward_shaping import screen_local_delta_reward

        prev = _make_synthetic_state("map")
        curr = _make_synthetic_state("map")
        curr["player"]["hp"] = prev["player"]["hp"] - 5

        # map is now in LOCAL_DELTA_SCREENS, HP loss should produce negative delta
        assert screen_local_delta_reward(prev, curr, "map") <= 0.0

    def test_combat_local_tactical_reward_prefers_vulnerable_setup(self):
        from rl_reward_shaping import combat_local_tactical_reward

        state = {
            "state_type": "monster",
            "player": {"hp": 60, "max_hp": 80, "block": 0, "energy": 2},
            "battle": {
                "energy": 2,
                "hand": [
                    {"id": "bash", "name": "Bash", "cost": 2, "damage": 8, "type": "ATTACK"},
                    {"id": "strike_ironclad", "name": "Strike", "cost": 1, "damage": 6, "type": "ATTACK"},
                ],
                "enemies": [
                    {
                        "entity_id": "jaw_0",
                        "name": "Jaw Worm",
                        "hp": 30,
                        "max_hp": 30,
                        "status": [],
                    }
                ],
            },
        }
        legal = [
            {"action": "play_card", "card_index": 0, "target": "jaw_0"},
            {"action": "play_card", "card_index": 1, "target": "jaw_0"},
            {"action": "end_turn"},
        ]

        assert combat_local_tactical_reward(state, legal[0], legal) > 0.0
        assert combat_local_tactical_reward(state, legal[1], legal) < 0.0

    def test_combat_local_tactical_reward_penalizes_zero_block_body_slam(self):
        from rl_reward_shaping import combat_local_tactical_reward

        state = {
            "state_type": "monster",
            "player": {"hp": 60, "max_hp": 80, "block": 0, "energy": 2},
            "battle": {
                "energy": 2,
                "hand": [
                    {"id": "body_slam", "name": "Body Slam", "cost": 0, "damage": 0, "type": "ATTACK"},
                    {"id": "defend_ironclad", "name": "Defend", "cost": 1, "block": 5, "type": "SKILL"},
                ],
                "enemies": [
                    {
                        "entity_id": "slime_0",
                        "name": "Sludge",
                        "hp": 30,
                        "max_hp": 30,
                        "status": [],
                    }
                ],
            },
        }
        legal = [
            {"action": "play_card", "card_index": 0, "target": "slime_0"},
            {"action": "play_card", "card_index": 1},
            {"action": "end_turn"},
        ]

        assert combat_local_tactical_reward(state, legal[0], legal) < 0.0
        assert combat_local_tactical_reward(state, legal[1], legal) > 0.0


# ---------------------------------------------------------------------------
# 9. Rollout alignment test — reward must match (s, a, s') not (prev_s, s, a)
# ---------------------------------------------------------------------------

class TestRolloutAlignment:
    """Verify that PPO buffer entries have correctly aligned (state, action, reward).

    The reward for step t should be shaped_reward(state_t, state_{t+1}),
    i.e. the reward AFTER executing action_t from state_t.
    A common bug is using shaped_reward(state_{t-1}, state_t) which
    shifts rewards by one step and destroys the learning signal.
    """

    def test_ppo_reward_comes_after_action(self):
        """Simulate the non-combat PPO collection pattern from train_hybrid.py
        and verify reward is computed from (pre_action_state, post_action_state)."""
        from rl_reward_shaping import shaped_reward

        # Simulate 3 states: floor 1 → floor 2 → floor 3
        states = [
            {"state_type": "map", "run": {"floor": 1, "act": 1},
             "player": {"hp": 70, "max_hp": 80, "gold": 50,
                        "relics": [{"id": "x"}],
                        "deck": [{"id": "STRIKE", "upgrades": 0}] * 5,
                        "potions": []}},
            {"state_type": "map", "run": {"floor": 2, "act": 1},
             "player": {"hp": 65, "max_hp": 80, "gold": 70,
                        "relics": [{"id": "x"}],
                        "deck": [{"id": "STRIKE", "upgrades": 0}] * 5,
                        "potions": []}},
            {"state_type": "map", "run": {"floor": 3, "act": 1},
             "player": {"hp": 60, "max_hp": 80, "gold": 90,
                        "relics": [{"id": "x"}],
                        "deck": [{"id": "STRIKE", "upgrades": 0}] * 5,
                        "potions": []}},
        ]

        # Correct pattern: reward for action at state[i] = shaped(state[i], state[i+1])
        rewards_correct = []
        for i in range(len(states) - 1):
            r = shaped_reward(states[i], states[i + 1], 0.0, done=False)
            rewards_correct.append(r)

        # Wrong pattern (the bug): reward = shaped(state[i-1], state[i])
        rewards_wrong = []
        for i in range(1, len(states)):
            r = shaped_reward(states[i - 1], states[i], 0.0, done=False)
            rewards_wrong.append(r)

        # Both should be positive (floor increasing), but they correspond to
        # DIFFERENT actions. The test verifies the correct pattern is used
        # by checking reward[0] corresponds to transition state[0]→state[1]
        r_floor_1_to_2 = shaped_reward(states[0], states[1], 0.0, done=False)
        assert rewards_correct[0] == r_floor_1_to_2, \
            "Reward for action at state[0] must be shaped(state[0], state[1])"

        # Verify the reward is finite and non-zero (direction depends on HP change)
        assert np.isfinite(r_floor_1_to_2), \
            f"Floor 1→2 reward should be finite, got {r_floor_1_to_2}"
        assert r_floor_1_to_2 != 0.0, \
            "Floor 1→2 reward should not be exactly 0 (PBRS should produce a signal)"

    def test_pending_step_pattern(self):
        """Verify the pending-step write pattern produces correct alignment.

        This mirrors the actual code pattern in train_hybrid.py:
          1. observe state → NN inference → get (action, log_prob, value)
          2. save as pending
          3. act(action) → get next_state
          4. reward = shaped(state, next_state)
          5. buffer.add(state, action, reward, value)
        """
        from rl_reward_shaping import shaped_reward

        s0 = {"state_type": "map", "run": {"floor": 1, "act": 1},
              "player": {"hp": 70, "max_hp": 80, "gold": 50,
                         "relics": [{"id": "x"}],
                         "deck": [{"id": "STRIKE", "upgrades": 0}] * 5,
                         "potions": []}}
        s1 = {"state_type": "map", "run": {"floor": 2, "act": 1},
              "player": {"hp": 70, "max_hp": 80, "gold": 80,
                         "relics": [{"id": "x"}, {"id": "y"}],
                         "deck": [{"id": "STRIKE", "upgrades": 0}] * 5,
                         "potions": []}}

        # Step 1-2: observe s0, sample action, save pending with pre_state=s0
        pending = {"pre_state": s0, "action_idx": 0, "log_prob": -1.0, "value": 0.0}

        # Step 3: act → get s1
        next_state = s1

        # Step 4: reward from CORRECT transition
        reward = shaped_reward(pending["pre_state"], next_state, 0.0, done=False)

        # Verify: reward corresponds to s0→s1 (floor 1→2), should be positive
        assert reward > 0, f"Pending-step reward s0→s1 should be positive, got {reward}"

        # Verify: this is NOT the same as shaped(prev_prev, s0) which would be
        # the old buggy pattern
        reward_wrong = shaped_reward(s1, s0, 0.0, done=False)  # reversed
        assert reward != reward_wrong, "Reward should differ from reversed transition"


# ---------------------------------------------------------------------------
# 10. Schema contract tests — C#/Python field name alignment
# ---------------------------------------------------------------------------

class TestSchemaContract:
    """Verify Python encoders read the correct field names that C# sends.

    These tests catch field name mismatches between McpMod.StateBuilder.cs
    and Python encoders. Each test creates a minimal state dict using the
    SAME field names that C# actually sends, and verifies Python reads them.
    """

    def test_map_reads_next_options(self, vocab):
        """C# sends map.next_options, not available_next_nodes or paths."""
        from rl_encoder_v2 import build_structured_state

        state = _make_synthetic_state("map")
        state["map"] = {
            "next_options": [   # C# field name (McpMod.StateBuilder.cs:1317)
                {"index": 0, "type": "monster", "col": 0, "row": 1},
                {"index": 1, "type": "elite", "col": 1, "row": 1},
                {"index": 2, "type": "rest_site", "col": 2, "row": 1},
            ]
        }
        ss = build_structured_state(state, vocab)
        # At least one map node should be encoded (not all zeros)
        assert ss.map_node_mask.any(), \
            "Map nodes not read — Python may be looking for wrong field name (available_next_nodes?)"

    def test_extract_player_prefers_richer_nested_payload(self, vocab):
        """Godot map/menu states may have a lightweight top-level player and a richer nested copy."""
        from rl_encoder_v2 import build_structured_state

        state = {
            "state_type": "map",
            "run": {"floor": 0, "act": 1},
            "player": {  # lightweight top-level payload
                "hp": 80,
                "max_hp": 80,
                "gold": 99,
                "relics": [{"id": "BURNING_BLOOD"}],
                "potions": [],
            },
            "map": {
                "player": {
                    "hp": 80,
                    "max_hp": 80,
                    "gold": 99,
                    "deck": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "cost": 1, "rarity": "BASIC"}] * 5,
                    "relics": [{"id": "BURNING_BLOOD"}],
                    "potions": [],
                    "open_potion_slots": 3,
                },
                "next_options": [{"index": 0, "type": "monster", "col": 0, "row": 1}],
            },
        }
        ss = build_structured_state(state, vocab)
        assert ss.deck_mask.sum() == 5, \
            "Encoder used the lightweight top-level player instead of the richer nested map.player payload"

    def test_map_falls_back_to_legal_actions_and_point_type(self, vocab):
        """Some backends expose map choices only in legal_actions and use point_type instead of type."""
        from rl_encoder_v2 import build_structured_state, build_structured_actions, NODE_TYPE_TO_IDX

        state = {
            "state_type": "map",
            "run": {"floor": 0, "act": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck": [], "relics": [], "potions": []},
            "map": {"next_options": []},
            "legal_actions": [
                {"action": "choose_map_node", "index": 0, "col": 0, "row": 1, "label": "monster", "point_type": "monster"},
                {"action": "choose_map_node", "index": 1, "col": 2, "row": 1, "label": "elite", "point_type": "elite"},
            ],
        }
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, state["legal_actions"], vocab)
        assert ss.map_node_mask[:2].tolist() == [True, True], \
            "Map encoder failed to recover node choices from legal_actions fallback"
        assert ss.map_node_types[0] == NODE_TYPE_TO_IDX["monster"]
        assert ss.map_node_types[1] == NODE_TYPE_TO_IDX["elite"]
        assert sa.target_node_types[0] == NODE_TYPE_TO_IDX["monster"]
        assert sa.target_node_types[1] == NODE_TYPE_TO_IDX["elite"]

    def test_player_powers_reads_status(self, vocab):
        """C# sends player.status, not player.powers or player.buffs."""
        from combat_nn import build_combat_features

        state = _make_synthetic_state("combat")
        state["player"]["status"] = [   # C# field name (McpMod.StateBuilder.cs:356)
            {"id": "strength", "amount": 5},
            {"id": "vulnerable", "amount": 2},
        ]
        sf = build_combat_features(state, vocab)
        # Strength should be encoded (scalar index 10)
        assert sf["scalars"][10] > 0, \
            "Player strength not read — Python may be looking for 'powers' instead of 'status'"

    def test_enemy_powers_reads_status(self, vocab):
        """C# sends enemy.status, not enemy.powers."""
        from rl_encoder_v2 import _enemy_aux_features

        enemy = {
            "id": "JAW_WORM", "hp": 40, "max_hp": 44, "block": 0,
            "status": [   # C# field name (McpMod.StateBuilder.cs:477)
                {"id": "strength", "amount": 3},
            ],
            "intents": [{"type": "attack", "damage": 11, "hits": 1}],
        }
        feat = _enemy_aux_features(enemy)
        # Strength at index 10
        assert feat[10] > 0, \
            "Enemy strength not read — Python may be looking for 'powers' instead of 'status'"

    def test_enemy_intent_reads_nested_intents(self, vocab):
        """C# sends enemy.intents[].type, not flat intent_type."""
        from rl_encoder_v2 import _enemy_aux_features

        enemy = {
            "id": "CULTIST", "hp": 48, "max_hp": 48, "block": 0,
            "status": [],
            "intents": [   # C# field name (McpMod.StateBuilder.cs:486-508)
                {"type": "attack", "damage": 6, "hits": 1},
            ],
        }
        feat = _enemy_aux_features(enemy)
        assert feat[3] == 1.0, \
            "Enemy attack intent not read — Python may be looking for flat 'intent_type'"
        assert feat[7] > 0, \
            "Enemy intent damage not read — Python may be looking for flat 'intent_damage'"

    def test_enemy_id_reads_entity_id(self, vocab):
        """C# sends entity_id (e.g. 'JAW_WORM_0'), not id."""
        from combat_nn import build_combat_features

        state = _make_synthetic_state("combat")
        # Use entity_id like C# actually sends
        state["battle"]["enemies"] = [
            {"entity_id": "JAW_WORM_0", "hp": 40, "max_hp": 44, "block": 0,
             "status": [], "intents": [{"type": "Attack", "damage": 11}]},
        ]
        sf = build_combat_features(state, vocab)
        # enemy_ids should be non-zero (JAW_WORM should be in vocab)
        assert sf["enemy_ids"][0] > 0, \
            "Enemy ID not read — Python may be looking for 'id' instead of 'entity_id'"

    def test_enemy_id_strips_runtime_suffix_for_vocab_lookup(self, vocab):
        """Runtime combat ids like SHRINKER_BEETLE_0 should resolve to the base monster vocab entry."""
        from rl_encoder_v2 import _cached_monster_idx

        base_idx = _cached_monster_idx(vocab, "SHRINKER_BEETLE")
        suffixed_idx = _cached_monster_idx(vocab, "SHRINKER_BEETLE_0")
        assert base_idx > 1, "Expected SHRINKER_BEETLE to exist in monster vocab"
        assert suffixed_idx == base_idx, \
            "Combat instance suffix should be stripped before monster vocab lookup"

    def test_intent_buff_not_match_debuff(self, vocab):
        """'DebuffStrong' should be debuff=1, buff=0 (not both)."""
        from rl_encoder_v2 import _enemy_aux_features

        enemy = {"entity_id": "X", "hp": 40, "max_hp": 44, "block": 0,
                 "status": [], "intents": [{"type": "DebuffStrong"}]}
        feat = _enemy_aux_features(enemy)
        assert feat[5] == 0.0, "DebuffStrong should NOT set is_buff flag"
        assert feat[6] == 1.0, "DebuffStrong should set is_debuff flag"

    def test_action_types_cover_all_screens(self):
        """All game action types should have a dedicated embedding index."""
        from rl_encoder_v2 import ACTION_TYPE_TO_IDX
        required = [
            "play_card", "end_turn", "choose_map_node", "select_card_reward",
            "choose_rest_option", "choose_event_option", "shop_purchase",
            "proceed", "claim_reward", "select_card", "confirm_selection",
            "cancel_selection", "skip",
        ]
        for action in required:
            assert action in ACTION_TYPE_TO_IDX, \
                f"Action type '{action}' missing from ACTION_TYPES — will fall through to 'other'"

    def test_screen_types_cover_all_states(self):
        """All game screen types should have a dedicated encoding index."""
        from rl_encoder_v2 import SCREEN_TYPE_TO_IDX
        required = [
            "combat", "monster", "elite", "boss", "map", "card_reward",
            "rest_site", "shop", "event", "card_select", "relic_select",
        ]
        for screen in required:
            assert screen in SCREEN_TYPE_TO_IDX, \
                f"Screen type '{screen}' missing — will fall through to 'other'"

    def test_node_type_index_zero_is_padding(self):
        """Index 0 must be 'unknown' (padding), not a real node type.

        rl_policy_v2.py uses `target_node_types > 0` to detect whether a
        node target exists. If a real type (e.g. 'monster') is at index 0,
        it gets silently filtered out.
        """
        from rl_encoder_v2 import NODE_TYPES
        assert NODE_TYPES[0] == "unknown", \
            f"NODE_TYPES[0] must be 'unknown' (padding), got '{NODE_TYPES[0]}'"


# ---------------------------------------------------------------------------
# 11. Feature non-zero assertions — catch silent data loss
# ---------------------------------------------------------------------------

class TestFeatureNonZero:
    """Verify that encoders produce non-zero features for realistic states.

    If a feature group is all-zero for a state where it should have data,
    it means the encoder is reading the wrong field name or missing data.
    """

    def test_map_node_features_nonzero(self, vocab):
        """Map state with next_options should produce non-zero node features."""
        from rl_encoder_v2 import build_structured_state
        state = {
            "state_type": "map",
            "run": {"floor": 3, "act": 1},
            "player": {"hp": 60, "max_hp": 80, "gold": 100,
                        "deck": [{"id": "STRIKE", "upgrades": 0}] * 5,
                        "relics": [{"id": "BURNING_BLOOD"}], "potions": []},
            "map": {
                "next_options": [
                    {"index": 0, "type": "monster", "col": 0, "row": 4},
                    {"index": 1, "type": "elite", "col": 1, "row": 4},
                ]
            },
        }
        ss = build_structured_state(state, vocab)
        assert ss.map_node_mask.any(), "Map node mask all False — next_options not read"
        assert ss.map_node_types.max() > 0, "Map node types all 0 — node type encoding broken"

    def test_combat_player_powers_nonzero(self, vocab):
        """Combat state with player status should encode powers."""
        from combat_nn import build_combat_features
        state = _make_synthetic_state("combat")
        state["player"]["status"] = [
            {"id": "strength", "amount": 3},
            {"id": "weak", "amount": 2},
        ]
        sf = build_combat_features(state, vocab)
        assert sf["scalars"][10] != 0, "Player strength not encoded (index 10)"
        assert sf["scalars"][13] != 0, "Player weak not encoded (index 13)"

    def test_combat_pile_counts_accept_numeric_counts_without_lists(self, vocab):
        """Headless can expose draw/discard/exhaust counts directly without full pile arrays."""
        from combat_nn import build_combat_features
        state = _make_synthetic_state("combat")
        state["player"].update({
            "draw_pile_count": 7,
            "discard_pile_count": 3,
            "exhaust_pile_count": 1,
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
        })
        state["battle"].pop("draw_pile_count", None)
        state["battle"].pop("discard_pile_count", None)
        state["battle"].pop("exhaust_pile_count", None)
        sf = build_combat_features(state, vocab)
        assert sf["scalars"][6] > 0, "draw_pile_count fallback not encoded"
        assert sf["scalars"][7] > 0, "discard_pile_count fallback not encoded"
        assert sf["scalars"][8] > 0, "exhaust_pile_count fallback not encoded"

    def test_combat_enemy_intent_nonzero(self, vocab):
        """Enemy with attack intent should have non-zero intent features."""
        from rl_encoder_v2 import _enemy_aux_features
        enemy = {
            "id": "JAW_WORM", "hp": 40, "max_hp": 44, "block": 0,
            "status": [],
            "intents": [{"type": "attack", "damage": 11, "hits": 1}],
        }
        feat = _enemy_aux_features(enemy)
        assert feat[3] == 1.0, "Attack intent flag not set (index 3)"
        assert feat[7] > 0, "Intent damage not encoded (index 7)"

    def test_combat_enemy_powers_nonzero(self, vocab):
        """Enemy with status should have non-zero power features."""
        from rl_encoder_v2 import _enemy_aux_features
        enemy = {
            "id": "CULTIST", "hp": 48, "max_hp": 48, "block": 0,
            "status": [{"id": "strength", "amount": 5}],
            "intents": [{"type": "buff"}],
        }
        feat = _enemy_aux_features(enemy)
        assert feat[10] > 0, "Enemy strength not encoded (index 10)"

    def test_ppo_scalars_nonzero_for_typical_state(self, vocab):
        """A mid-run state should have non-zero scalar features."""
        from rl_encoder_v2 import build_structured_state
        state = {
            "state_type": "map",
            "run": {"floor": 5, "act": 1},
            "player": {"hp": 50, "max_hp": 80, "gold": 120, "energy": 3,
                        "deck": [{"id": "STRIKE", "upgrades": 1}] * 3 +
                                [{"id": "DEFEND_IRONCLAD", "upgrades": 0}] * 3,
                        "relics": [{"id": "BURNING_BLOOD"}, {"id": "VAJRA"}],
                        "potions": [{"id": "FIRE_POTION"}]},
            "map": {"next_options": [{"index": 0, "type": "monster"}]},
        }
        ss = build_structured_state(state, vocab)
        # HP, gold, floor should all be non-zero
        assert ss.scalars[0] > 0, "HP ratio should be > 0"
        assert ss.deck_mask.sum() > 0, "Deck should have cards"
        assert ss.relic_mask.sum() > 0, "Should have relics"

    def test_noncombat_scalars_ignore_leaked_combat_energy_fields(self, vocab):
        """Map/event screens should not drift because one backend leaks max_energy/player block."""
        from rl_encoder_v2 import build_structured_state
        state = {
            "state_type": "map",
            "run": {"floor": 0, "act": 1},
            "player": {
                "hp": 80,
                "max_hp": 80,
                "gold": 99,
                "energy": 0,
                "max_energy": 3,
                "block": 12,
                "deck": [{"id": "STRIKE_IRONCLAD", "type": "ATTACK", "cost": 1, "rarity": "BASIC"}] * 5,
                "relics": [{"id": "BURNING_BLOOD"}],
                "potions": [],
            },
            "legal_actions": [{"action": "choose_map_node", "index": 0, "label": "monster"}],
        }
        ss = build_structured_state(state, vocab)
        assert ss.scalars[5] == 0.0, "Non-combat map state should ignore leaked current energy"
        assert ss.scalars[6] == 0.0, "Non-combat map state should ignore leaked max energy"
        assert ss.scalars[7] == 0.0, "Non-combat map state should ignore leaked block"
        assert ss.scalars[8] == 0.0, "Non-combat map state should ignore leaked round number"


# ---------------------------------------------------------------------------
# 12. Action distinguishability — different options must produce different repr
# ---------------------------------------------------------------------------

class TestActionDistinguishability:
    """Verify that different action choices produce different feature vectors.

    If two options produce identical features, the NN cannot tell them apart
    and is effectively choosing randomly between them.
    """

    def test_map_nodes_distinguishable(self, vocab):
        """Different map node types should produce different action features."""
        from rl_encoder_v2 import build_structured_actions
        state = {
            "state_type": "map",
            "map": {"next_options": [
                {"index": 0, "type": "monster", "col": 0, "row": 1},
                {"index": 1, "type": "elite", "col": 1, "row": 1},
                {"index": 2, "type": "rest_site", "col": 2, "row": 1},
            ]},
            "player": {"hp": 60, "max_hp": 80},
        }
        actions = [
            {"action": "choose_map_node", "index": 0},
            {"action": "choose_map_node", "index": 1},
            {"action": "choose_map_node", "index": 2},
        ]
        sa = build_structured_actions(state, actions, vocab)
        # Each action should have a different target_node_type
        assert sa.target_node_types[0] != sa.target_node_types[1], \
            "Monster and elite map nodes have same encoding — indistinguishable"
        assert sa.target_node_types[1] != sa.target_node_types[2], \
            "Elite and rest_site map nodes have same encoding — indistinguishable"

    def test_card_rewards_distinguishable(self, vocab):
        """Different card reward choices should produce different action features."""
        from rl_encoder_v2 import build_structured_actions
        state = {
            "state_type": "card_reward",
            "card_reward": {"cards": [
                {"id": "STRIKE", "name": "Strike"},
                {"id": "POMMEL_STRIKE", "name": "Pommel Strike"},
                {"id": "SHRUG_IT_OFF", "name": "Shrug It Off"},
            ]},
            "player": {"hp": 60, "max_hp": 80},
        }
        actions = [
            {"action": "select_card_reward", "index": 0},
            {"action": "select_card_reward", "index": 1},
            {"action": "select_card_reward", "index": 2},
        ]
        sa = build_structured_actions(state, actions, vocab)
        # Different cards should have different target_card_ids
        # (at least STRIKE vs POMMEL_STRIKE should differ)
        assert sa.target_card_ids[0] != sa.target_card_ids[1], \
            "Different card rewards have same card ID encoding — indistinguishable"

    def test_event_options_distinguishable(self, vocab):
        """Different event options should produce different action features."""
        from rl_encoder_v2 import build_structured_actions
        state = {"state_type": "event", "player": {"hp": 60, "max_hp": 80}}
        actions = [
            {"action": "choose_event_option", "index": 0},
            {"action": "choose_event_option", "index": 1},
            {"action": "choose_event_option", "index": 2},
        ]
        sa = build_structured_actions(state, actions, vocab)
        # target_indices should differ
        assert sa.target_indices[0] != sa.target_indices[1], \
            "Event option 0 and 1 have same index — indistinguishable"

    def test_rest_options_distinguishable(self, vocab):
        """Rest and smith should produce different action features."""
        from rl_encoder_v2 import build_structured_actions, ACTION_TYPE_TO_IDX
        state = {"state_type": "rest_site", "player": {"hp": 60, "max_hp": 80}}
        actions = [
            {"action": "choose_rest_option", "index": 0},  # rest
            {"action": "choose_rest_option", "index": 1},  # smith
        ]
        sa = build_structured_actions(state, actions, vocab)
        # Same action type but different index
        assert sa.target_indices[0] != sa.target_indices[1], \
            "Rest and smith have same target_index — indistinguishable"


# ---------------------------------------------------------------------------
# GPT-Design Optimization Tests
# ---------------------------------------------------------------------------

class TestScreenValueHeads:
    """Phase 1A: Screen-specific value heads."""

    def test_network_has_value_heads(self, vocab):
        from rl_policy_v2 import FullRunPolicyNetworkV2
        net = FullRunPolicyNetworkV2(vocab=vocab)
        assert hasattr(net, "value_heads")
        assert "combat" in net.value_heads
        assert "map" in net.value_heads
        assert "card_reward" in net.value_heads
        assert "campfire" in net.value_heads
        assert "shop" in net.value_heads
        assert "event" in net.value_heads
        assert "default" in net.value_heads

    def test_different_screens_use_different_heads(self, vocab):
        from rl_policy_v2 import FullRunPolicyNetworkV2, _structured_state_to_numpy_dict, _structured_actions_to_numpy_dict
        from rl_encoder_v2 import build_structured_state, build_structured_actions

        net = FullRunPolicyNetworkV2(vocab=vocab)
        net.eval()

        # Test with map screen
        state_map = _make_synthetic_state("map")
        actions_map = _make_synthetic_actions("map")
        ss_map = build_structured_state(state_map, vocab)
        sa_map = build_structured_actions(state_map, actions_map, vocab)
        st_map = {k: (torch.tensor(v).unsqueeze(0).long() if ("ids" in k or "idx" in k or "types" in k or "count" in k)
                       else torch.tensor(v).unsqueeze(0).bool() if "mask" in k
                       else torch.tensor(v).unsqueeze(0).float()) if isinstance(v, np.ndarray)
                  else (torch.tensor([v]).long() if ("ids" in k or "idx" in k or "types" in k or "count" in k)
                        else torch.tensor([v]))
                  for k, v in _structured_state_to_numpy_dict(ss_map).items()}
        at_map = {k: (torch.tensor(v).unsqueeze(0).long() if ("ids" in k or "types" in k or "indices" in k)
                       else torch.tensor(v).unsqueeze(0).bool() if "mask" in k
                       else torch.tensor(v).unsqueeze(0).float()) if isinstance(v, np.ndarray)
                  else torch.tensor([v])
                  for k, v in _structured_actions_to_numpy_dict(sa_map).items()}

        with torch.no_grad():
            logits, values, _, boss_ready, action_adv = net.forward(st_map, at_map)
        assert torch.isfinite(values).all()
        assert values.shape == (1,)
        assert torch.isfinite(boss_ready).all()
        assert torch.isfinite(action_adv).all()


class TestPerScreenAdvNorm:
    """Phase 1B: Per-screen advantage normalization."""

    def test_normalize_per_screen(self):
        from rl_policy_v2 import PPOTrainerV2
        # 10 steps: 5 from screen 4 (map), 5 from screen 5 (card_reward)
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0,  # map (high)
                            0.1, 0.2, 0.3, 0.4, 0.5])  # card_reward (low)
        screen_idx = torch.tensor([4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        result = PPOTrainerV2._normalize_advantages_per_screen(adv, screen_idx)
        # Map group should be normalized within itself
        map_adv = result[:5]
        card_adv = result[5:]
        assert abs(map_adv.mean().item()) < 0.01
        assert abs(card_adv.mean().item()) < 0.01

    def test_small_group_fallback(self):
        from rl_policy_v2 import PPOTrainerV2
        # 8 steps: 6 from map, 2 from shop (too few for per-screen)
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,  # map
                            10.0, 20.0])  # shop (only 2 — below threshold)
        screen_idx = torch.tensor([4, 4, 4, 4, 4, 4, 7, 7])
        result = PPOTrainerV2._normalize_advantages_per_screen(adv, screen_idx, min_group_size=4)
        # Map group should be normalized
        assert abs(result[:6].mean().item()) < 0.01
        # Shop group uses global normalization (not per-screen)
        assert torch.isfinite(result).all()


class TestSegmentBuffer:
    """Phase 2: Semi-MDP segment buffer."""

    def test_segment_gae_basic(self):
        from rl_segment_buffer import SegmentRolloutBuffer, Segment
        buf = SegmentRolloutBuffer()
        # 3 segments: short, long, terminal
        buf.add(Segment(state={}, actions={}, action_idx=0, log_prob=-1.0,
                        value=0.5, reward_sum=0.1, seg_len=1, done=False, screen_type_idx=4))
        buf.add(Segment(state={}, actions={}, action_idx=1, log_prob=-1.0,
                        value=0.6, reward_sum=0.5, seg_len=10, done=False, screen_type_idx=5))
        buf.add(Segment(state={}, actions={}, action_idx=0, log_prob=-1.0,
                        value=0.4, reward_sum=1.0, seg_len=5, done=True, screen_type_idx=4))
        buf.compute_gae()
        assert len(buf.advantages) == 3
        assert all(isinstance(a, float) for a in buf.advantages)
        # Terminal segment: advantage = reward - value (since done=True, next_value=0)
        # For seg_len=5: discount = 0.999^5 ≈ 0.995
        # delta = 1.0 + 0 - 0.4 = 0.6
        assert buf.advantages[2] > 0

    def test_segment_gae_discount_capping(self):
        from rl_segment_buffer import SegmentRolloutBuffer, Segment
        buf = SegmentRolloutBuffer()
        buf.add(Segment(state={}, actions={}, action_idx=0, log_prob=-1.0,
                        value=0.5, reward_sum=0.1, seg_len=100, done=True, screen_type_idx=4))
        buf.compute_gae(max_discount_steps=32)
        # seg_len=100 is capped to 32
        assert len(buf.advantages) == 1

    def test_segment_stats(self):
        from rl_segment_buffer import SegmentRolloutBuffer, Segment
        buf = SegmentRolloutBuffer()
        buf.add(Segment(state={}, actions={}, action_idx=0, log_prob=-1.0,
                        value=0.5, reward_sum=0.1, seg_len=3, done=False, screen_type_idx=4))
        buf.add(Segment(state={}, actions={}, action_idx=0, log_prob=-1.0,
                        value=0.5, reward_sum=0.5, seg_len=10, done=True, screen_type_idx=5))
        stats = buf.get_segment_stats()
        assert stats["num_segments"] == 2
        assert stats["seg_len_mean"] == 6.5


class TestSegmentCollector:
    """Phase 2B: Segment collector."""

    def test_open_close(self):
        from segment_collector import NonCombatSegmentCollector
        col = NonCombatSegmentCollector()
        assert not col.is_open
        col.open_segment(state={}, actions={}, action_idx=0, log_prob=-1.0,
                         value=0.5, screen_type_idx=4)
        assert col.is_open
        col.add_reward(0.1, tag="pbrs", steps=1)
        col.add_reward(0.2, tag="fight_summary", steps=5)
        seg = col.close_segment(done=False)
        assert not col.is_open
        assert seg is not None
        assert abs(seg.reward_sum - 0.3) < 1e-6
        assert seg.seg_len == 6

    def test_close_without_open(self):
        from segment_collector import NonCombatSegmentCollector
        col = NonCombatSegmentCollector()
        seg = col.close_segment(done=False)
        assert seg is None


class TestCounterfactualScoring:
    """Phase 3: Screen-local counterfactual scoring."""

    def test_card_reward_scoring(self):
        from counterfactual_scoring import score_card_reward
        state = _make_synthetic_state("card_reward")
        state["card_reward"] = {"cards": [
            {"id": "POMMEL_STRIKE", "cost": 1, "type": "ATTACK", "upgrades": 0},
        ]}
        actions = [
            {"action": "select_card_reward", "index": 0,
             "card": {"id": "POMMEL_STRIKE", "cost": 1, "type": "ATTACK", "upgrades": 0}},
            {"action": "skip"},
        ]
        scores = score_card_reward(state, actions)
        assert len(scores) == 2
        # Adding a card should generally change the score (not necessarily better)
        assert isinstance(scores[0], float)
        assert isinstance(scores[1], float)

    def test_counterfactual_reward(self):
        from counterfactual_scoring import counterfactual_reward
        scores = [0.1, 0.05, 0.0, -0.03]
        # Choosing the best should give positive reward
        reward_best, teacher = counterfactual_reward(0, scores)
        reward_worst, _ = counterfactual_reward(3, scores)
        assert reward_best >= reward_worst

    def test_dispersion_guard(self):
        from counterfactual_scoring import counterfactual_reward
        # All scores nearly identical — should return 0
        scores = [0.001, 0.001, 0.001]
        reward, teacher = counterfactual_reward(0, scores, min_dispersion=0.01)
        assert reward == 0.0
        assert teacher is None

    def test_teacher_distribution_valid(self):
        from counterfactual_scoring import counterfactual_reward
        scores = [0.3, 0.1, -0.1, 0.0]
        _, teacher = counterfactual_reward(0, scores)
        assert teacher is not None
        assert abs(sum(teacher) - 1.0) < 0.01
        assert all(p >= 0 for p in teacher)

    def test_gold_not_directly_rewarded(self):
        """Gold holding should not produce positive reward by itself."""
        from rl_reward_shaping import economy_score
        # economy_score returns threshold-based utility, not raw gold
        assert economy_score({"player": {"gold": 0}}) == 0.0
        # But having gold near a threshold gives value
        assert economy_score({"player": {"gold": 75}}) > 0

    def test_elite_clear_positive_with_moderate_hp_loss(self):
        """Winning elite fight with moderate HP loss should still be positive."""
        from rl_reward_shaping import fight_summary
        # 20% HP loss on elite (expected is 30%), so no excess
        result = fight_summary(hp_before=80, hp_after=64, max_hp=80, won=True, room_type="elite")
        assert result > 0, f"Elite clear with moderate HP loss gave negative reward: {result}"

    def test_skip_not_punished(self):
        """Skip on bad card reward should not be negative."""
        from counterfactual_scoring import score_card_reward
        state = _make_synthetic_state("card_reward")
        # Offer only strikes to an already strike-heavy deck
        state["player"]["deck"] = [
            {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "ATTACK", "upgrades": 0},
        ] * 10
        state["card_reward"] = {"cards": [
            {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "ATTACK", "upgrades": 0},
        ]}
        actions = [
            {"action": "select_card_reward", "index": 0,
             "card": {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "ATTACK", "upgrades": 0}},
            {"action": "skip"},
        ]
        scores = score_card_reward(state, actions)
        # Skip (scores[1]) should be >= taking another strike (scores[0])
        # when deck is already bloated with strikes
        assert scores[1] >= 0, "Skip has negative score"

    def test_map_scoring(self):
        from counterfactual_scoring import score_map_choice
        state = _make_synthetic_state("map")
        actions = [
            {"action": "choose_map_node", "node_type": "monster"},
            {"action": "choose_map_node", "node_type": "rest_site"},
            {"action": "choose_map_node", "node_type": "elite"},
        ]
        scores = score_map_choice(state, actions)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)


class TestGenericStateDelta:
    """Phase 3 prerequisite: screen-specific state delta."""

    def test_screen_mix_exists(self):
        from rl_reward_shaping import _SCREEN_MIX
        assert "card_reward" in _SCREEN_MIX
        assert "campfire" in _SCREEN_MIX
        assert "map" in _SCREEN_MIX
        assert "shop" in _SCREEN_MIX

    def test_card_reward_problem_heavy(self):
        from rl_reward_shaping import _SCREEN_MIX
        w_p, w_s, w_e = _SCREEN_MIX["card_reward"]
        assert w_p > w_s, "card_reward should weight problem more than survival"
        assert w_p > w_e, "card_reward should weight problem more than economy"

    def test_campfire_survival_significant(self):
        from rl_reward_shaping import _SCREEN_MIX
        w_p, w_s, w_e = _SCREEN_MIX["campfire"]
        assert w_s > w_e, "campfire should weight survival more than economy"
        assert w_s >= 0.35, "campfire survival weight should be significant"


class TestTrainingConfig:
    def test_counterfactual_weight_without_scoring_is_disabled(self):
        from train_hybrid import _resolve_counterfactual_runtime

        effective_scoring, effective_weight, warnings = _resolve_counterfactual_runtime(
            use_segment_collector=False,
            counterfactual_scoring=False,
            counterfactual_weight=0.4,
        )

        assert effective_scoring is False
        assert effective_weight == 0.0
        assert any("effective counterfactual weight is 0.0" in msg for msg in warnings)

    def test_counterfactual_requires_segment_collector(self):
        from train_hybrid import _resolve_counterfactual_runtime

        effective_scoring, effective_weight, warnings = _resolve_counterfactual_runtime(
            use_segment_collector=False,
            counterfactual_scoring=True,
            counterfactual_weight=0.2,
        )

        assert effective_scoring is False
        assert effective_weight == 0.0
        assert any("requires --use-segment-collector" in msg for msg in warnings)


class TestCombatCurriculum:
    def test_parse_room_types_filters_unknown_tokens(self):
        _parse_room_types = _import_or_skip("train_combat_only")._parse_room_types

        assert _parse_room_types("boss,elite,unknown") == {"boss", "elite"}

    def test_should_train_combat_respects_floor_and_room_type(self):
        _should_train_combat = _import_or_skip("train_combat_only")._should_train_combat

        assert _should_train_combat(16, "boss", 14, {"boss"}) is True
        assert _should_train_combat(12, "boss", 14, {"boss"}) is False
        assert _should_train_combat(16, "monster", 14, {"boss"}) is False

    def test_load_seeds_from_json_array(self, tmp_path):
        _load_seeds_from_file = _import_or_skip("train_combat_only")._load_seeds_from_file

        seed_path = tmp_path / "seeds.json"
        seed_path.write_text('["seed-a", "seed-b"]', encoding="utf-8")

        assert _load_seeds_from_file(str(seed_path)) == ["seed-a", "seed-b"]

    def test_load_seeds_from_benchmark_manifest_prefers_benchmark(self, tmp_path):
        _load_seeds_from_file = _import_or_skip("train_combat_only")._load_seeds_from_file

        seed_path = tmp_path / "seed_manifest.json"
        seed_path.write_text(
            json.dumps({
                "smoke": [{"seed": "smoke-01"}],
                "benchmark": [{"seed": "bench-01"}, {"seed": "bench-02"}],
            }),
            encoding="utf-8",
        )

        assert _load_seeds_from_file(str(seed_path)) == ["bench-01", "bench-02"]

    def test_cyclic_seed_source_repeats_in_order(self):
        CyclicSeedSource = _import_or_skip("train_combat_only").CyclicSeedSource

        seed_source = CyclicSeedSource(["s1", "s2"])

        assert [seed_source.next_seed() for _ in range(5)] == ["s1", "s2", "s1", "s2", "s1"]


class TestEvaluateHarness:
    def test_archive_wrappers_reexport_symbols(self):
        specialist = _import_or_skip("build_act1_specialist_datasets")
        export_mod = _import_or_skip("export_combat_training_data_from_full_run")
        train_bc = _import_or_skip("train_behavior_clone")
        eval_bc = _import_or_skip("evaluate_behavior_clone")

        assert callable(specialist.main)
        assert callable(export_mod.export_paths)
        assert callable(train_bc.main)
        assert callable(eval_bc.main)

    def test_headless_sim_runner_defaults_point_at_workspace(self):
        import headless_sim_runner

        repo_root_expected = Path(__file__).resolve().parents[2]
        repo_root = Path(headless_sim_runner.DEFAULT_REPO_ROOT).resolve()
        dll_path = Path(headless_sim_runner.DEFAULT_DLL_PATH).resolve()
        expected_candidates = [
            repo_root_expected / "STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe",
            repo_root_expected / "STS2AI/ENV/Sim/Runtime/HeadlessSim/bin/Debug/net9.0/HeadlessSim.dll",
            repo_root_expected / "STS2AI/overlay/headless-sim-host/bin/Debug/net9.0/headless_sim_host_0991.exe",
            repo_root_expected / "STS2AI/tools/headless-sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.dll",
            repo_root_expected / "overlay/headless-sim-host/bin/Debug/net9.0/headless_sim_host_0991.exe",
            repo_root_expected / "tools/headless-sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.dll",
        ]

        assert repo_root == repo_root_expected
        assert dll_path in {candidate.resolve() for candidate in expected_candidates}

    def test_verify_save_load_defaults_point_at_workspace(self):
        import verify_save_load

        repo_root_expected = Path(__file__).resolve().parents[2]
        repo_root = Path(verify_save_load.DEFAULT_REPO_ROOT).resolve()
        dll_path = Path(verify_save_load.DEFAULT_HEADLESS_DLL).resolve()
        expected_candidates = [
            repo_root_expected / "STS2AI/ENV/Sim/Host/bin/Debug/net9.0/headless_sim_host_0991.exe",
            repo_root_expected / "STS2AI/ENV/Sim/Runtime/HeadlessSim/bin/Debug/net9.0/HeadlessSim.dll",
            repo_root_expected / "STS2AI/overlay/headless-sim-host/bin/Debug/net9.0/headless_sim_host_0991.exe",
            repo_root_expected / "STS2AI/tools/headless-sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.dll",
            repo_root_expected / "overlay/headless-sim-host/bin/Debug/net9.0/headless_sim_host_0991.exe",
            repo_root_expected / "tools/headless-sim/HeadlessSim/bin/Debug/net9.0/HeadlessSim.dll",
        ]

        assert repo_root == repo_root_expected
        assert dll_path in {candidate.resolve() for candidate in expected_candidates}

    def test_checkpoint_dim_inference_prefers_weight_shapes(self):
        from evaluate_ai import _infer_ppo_embed_dim, _infer_combat_dims

        ppo_state = {
            "entity_emb.card_embed.weight": torch.zeros((579, 48)),
        }
        combat_state = {
            "entity_emb.card_embed.weight": torch.zeros((579, 48)),
            "action_proj.weight": torch.zeros((192, 144)),
        }

        assert _infer_ppo_embed_dim(ppo_state, fallback=32) == 48
        assert _infer_combat_dims(combat_state, fallback_embed_dim=32, fallback_hidden_dim=128) == (48, 192)

    def test_combat_rewards_auto_progress_claims_then_proceeds(self):
        from evaluate_ai import _choose_auto_progress_action

        state = {"state_type": "combat_rewards"}
        claim_first = [
            {"action": "claim_reward", "label": "Take Gold"},
            {"action": "proceed", "label": "Continue"},
        ]
        proceed_only = [
            {"action": "proceed", "label": "Continue"},
        ]

        assert _choose_auto_progress_action(state, "combat_rewards", claim_first)["action"] == "claim_reward"
        assert _choose_auto_progress_action(state, "combat_rewards", proceed_only)["action"] == "proceed"

    def test_combat_rewards_auto_progress_repeated_claim_prefers_proceed(self):
        from evaluate_ai import _choose_auto_progress_action, _reward_claim_signature

        legal = [
            {"action": "claim_reward", "label": "DUPLICATOR"},
            {"action": "proceed", "label": "Continue"},
        ]
        state = {"state_type": "combat_rewards", "legal_actions": legal}

        claim_sig = _reward_claim_signature(state, legal[0])
        chosen = _choose_auto_progress_action(state, "combat_rewards", legal, claim_sig)
        assert chosen is not None
        assert chosen["action"] == "proceed"

    def test_combat_rewards_auto_progress_skips_unclaimable_potion(self):
        from evaluate_ai import _choose_auto_progress_action

        state = {
            "state_type": "combat_rewards",
            "player": {"open_potion_slots": 0},
            "rewards": {
                "player": {"open_potion_slots": 0},
                "items": [
                    {"index": 0, "type": "potion", "label": "DUPLICATOR"},
                    {"index": 1, "type": "gold", "label": "Gold"},
                ],
                "can_proceed": True,
            },
        }
        legal = [
            {"action": "claim_reward", "index": 0, "label": "DUPLICATOR"},
            {"action": "claim_reward", "index": 1, "label": "Gold"},
            {"action": "proceed", "label": "Continue"},
        ]

        chosen = _choose_auto_progress_action(state, "combat_rewards", legal)
        assert chosen is not None
        assert chosen["action"] == "claim_reward"
        assert chosen["index"] == 1

    def test_repeat_loop_tracker_triggers_escape(self):
        from evaluate_ai import RepeatLoopTracker

        legal = [
            {"action": "choose_event_option", "label": "Greed"},
            {"action": "proceed", "label": "Leave"},
        ]
        tracker = RepeatLoopTracker(trigger_count=2, max_repeats=5)

        assert tracker.choose_escape_action(legal) is None
        tracker.observe("event", legal)
        tracker.observe("event", legal)
        tracker.observe("event", legal)

        escape = tracker.choose_escape_action(legal)
        assert escape is not None
        assert escape["action"] == "proceed"

    def test_train_hybrid_auto_progress_proceeds_after_card_reward_into_event(self):
        from train_hybrid import _choose_auto_progress_action

        state = {
            "state_type": "event",
            "event": {
                "event_id": "EVENT.DENSE_VEGETATION",
                "in_dialogue": False,
                "is_finished": True,
            },
        }
        legal = [
            {"action": "proceed", "label": "Leave"},
        ]

        chosen = _choose_auto_progress_action(state, legal, last_action_name="skip_card_reward")
        assert chosen is not None
        assert chosen["action"] == "proceed"

    def test_train_hybrid_combat_rewards_repeated_claim_before_card_reward_keeps_claiming(self):
        from train_hybrid import _choose_auto_progress_action, _reward_claim_signature

        state = {"state_type": "combat_rewards"}
        legal = [
            {"action": "claim_reward", "label": "DUPLICATOR", "index": 0},
            {"action": "proceed", "label": "Continue"},
        ]

        claim_sig = _reward_claim_signature(state | {"legal_actions": legal}, legal[0])
        chosen = _choose_auto_progress_action(
            state,
            legal,
            last_action_name="claim_reward",
            last_reward_claim_sig=claim_sig,
        )
        assert chosen is not None
        assert chosen["action"] == "claim_reward"

    def test_train_hybrid_combat_rewards_repeated_claim_after_card_reward_prefers_proceed(self):
        from train_hybrid import _choose_auto_progress_action, _reward_claim_signature

        state = {"state_type": "combat_rewards", "legal_actions": [
            {"action": "claim_reward", "label": "DUPLICATOR", "index": 0},
            {"action": "proceed", "label": "Continue"},
        ]}
        legal = list(state["legal_actions"])

        claim_sig = _reward_claim_signature(state, legal[0])
        chosen = _choose_auto_progress_action(
            state,
            legal,
            last_action_name="skip_card_reward",
            last_reward_claim_sig=claim_sig,
            last_reward_claim_count=1,
            reward_chain_card_reward_seen=True,
        )
        assert chosen is not None
        assert chosen["action"] == "proceed"

    def test_train_hybrid_combat_rewards_skips_unclaimable_potion(self):
        from train_hybrid import _choose_auto_progress_action

        state = {
            "state_type": "combat_rewards",
            "player": {"open_potion_slots": 0},
            "rewards": {
                "player": {"open_potion_slots": 0},
                "items": [
                    {"index": 0, "type": "potion", "label": "DUPLICATOR"},
                    {"index": 1, "type": "gold", "label": "Gold"},
                ],
                "can_proceed": True,
            },
        }
        legal = [
            {"action": "claim_reward", "index": 0, "label": "DUPLICATOR"},
            {"action": "claim_reward", "index": 1, "label": "Gold"},
            {"action": "proceed", "label": "Continue"},
        ]

        chosen = _choose_auto_progress_action(state, legal)
        assert chosen is not None
        assert chosen["action"] == "claim_reward"
        assert chosen["index"] == 1

    def test_train_hybrid_empty_legal_event_recovery_prefers_dialogue_then_proceed(self):
        from train_hybrid import _choose_empty_legal_recovery_action

        dialogue_state = {
            "state_type": "event",
            "event": {"in_dialogue": True, "is_finished": False},
        }
        proceed_state = {
            "state_type": "event",
            "event": {"in_dialogue": False, "is_finished": True, "can_proceed": True},
        }
        unsettled_post_reward_state = {
            "state_type": "event",
            "event": {"in_dialogue": False, "options": []},
        }
        settled_post_reward_state = {
            "state_type": "event",
            "event": {"in_dialogue": False, "options": [], "is_finished": True, "can_proceed": True},
        }

        assert _choose_empty_legal_recovery_action(dialogue_state) == {"action": "advance_dialogue"}
        assert _choose_empty_legal_recovery_action(proceed_state) == {"action": "proceed"}
        assert _choose_empty_legal_recovery_action(unsettled_post_reward_state, "skip_card_reward") is None
        assert _choose_empty_legal_recovery_action(settled_post_reward_state, "skip_card_reward") == {"action": "proceed"}

    def test_train_hybrid_topk_action_summary_accepts_torch_tensor(self):
        from train_hybrid import _topk_action_summary

        legal = [
            {"action": "play_card", "label": "Bash"},
            {"action": "play_card", "label": "Strike"},
            {"action": "end_turn", "label": "End Turn"},
        ]
        logits = torch.tensor([[2.0, 1.0, -3.0]], dtype=torch.float32)

        summary = _topk_action_summary(legal, logits, k=2)

        assert "Bash:" in summary
        assert "Strike:" in summary

    def test_evaluate_timeout_snapshot_separates_last_state_from_timeout(self):
        from evaluate_ai import GameResult, _record_last_state_snapshot, _record_timeout_snapshot

        result = GameResult()
        _record_last_state_snapshot(result, "monster", 7, 5)

        assert result.last_state_type == "monster"
        assert result.last_state_floor == 7
        assert result.last_state_legal_action_count == 5
        assert result.timeout_state_type == ""

        _record_timeout_snapshot(result)

        assert result.timeout_state_type == "monster"
        assert result.timeout_floor == 7
        assert result.timeout_legal_action_count == 5

    def test_evaluate_match_legal_action_index_prefers_raw_key(self):
        from evaluate_ai import _match_legal_action_index

        legal = [
            {"action": "play_card", "card_index": 0, "target": "enemy-0", "label": "Bash"},
            {"action": "end_turn", "label": "End Turn"},
        ]
        chosen = {"action": "play_card", "card_index": 0, "target": "enemy-0", "label": "Bash"}

        assert _match_legal_action_index(legal, chosen) == 0

    def test_make_trace_step_includes_combat_mcts_summary(self):
        from evaluate_ai import CombatMctsTrace, _make_trace_step

        state = _make_synthetic_state("monster")
        legal = [
            {"action": "play_card", "card_index": 0, "target": "enemy-0", "label": "Bash"},
            {"action": "end_turn", "label": "End Turn"},
        ]
        trace = CombatMctsTrace(
            chosen_action={"action": "play_card", "card_index": 0, "target": "enemy-0"},
            top_actions=[
                {
                    "action": {"action": "play_card", "card_index": 0, "target": "enemy-0"},
                    "prior": 0.6,
                    "visits": 12,
                    "visit_frac": 0.75,
                    "q": 0.4,
                },
                {
                    "action": {"action": "end_turn"},
                    "prior": 0.1,
                    "visits": 1,
                    "visit_frac": 0.0625,
                    "q": -0.2,
                },
            ],
            sims=16,
            root_value=0.25,
        )

        step = _make_trace_step(
            state,
            legal,
            chosen_action=legal[0],
            action_source="combat_mcts",
            step_index=3,
            combat_mcts_trace=trace,
        )

        assert step["combat_mcts"]["sims"] == 16
        assert step["combat_mcts"]["root_value"] == pytest.approx(0.25)
        assert step["combat_mcts"]["top_actions"][0]["action"]["action"] == "play_card"
        assert step["combat_mcts"]["top_actions"][0]["q"] == pytest.approx(0.4)

    def test_select_combat_teacher_index_prefers_score_head_without_probe(self):
        from evaluate_ai import _select_combat_teacher_index

        legal = [
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD"},
            {"action": "play_card", "card_id": "DEFEND_IRONCLAD"},
            {"action": "end_turn"},
        ]
        masked_scores = np.asarray([0.2, 0.8, -0.5], dtype=np.float32)
        masked_logits = np.asarray([1.5, 0.2, -1.0], dtype=np.float32)

        action_idx, source = _select_combat_teacher_index(
            legal=legal,
            masked_scores=masked_scores,
            masked_logits=masked_logits,
            lethal_logit_blend_alpha=0.5,
            direct_lethal_probe_top_k=4,
            direct_lethal_probe=lambda _: set(),
        )

        assert action_idx == 1
        assert source == "combat_teacher_scores"

    def test_select_combat_teacher_index_uses_direct_lethal_blend_when_probe_hits(self):
        from evaluate_ai import _select_combat_teacher_index

        legal = [
            {"action": "play_card", "card_id": "STRIKE_IRONCLAD"},
            {"action": "play_card", "card_id": "DEFEND_IRONCLAD"},
            {"action": "end_turn"},
        ]
        masked_scores = np.asarray([0.1, 1.2, -0.3], dtype=np.float32)
        masked_logits = np.asarray([2.4, 0.1, -1.0], dtype=np.float32)

        action_idx, source = _select_combat_teacher_index(
            legal=legal,
            masked_scores=masked_scores,
            masked_logits=masked_logits,
            lethal_logit_blend_alpha=0.5,
            direct_lethal_probe_top_k=4,
            direct_lethal_probe=lambda _: {0},
        )

        assert action_idx == 0
        assert source == "combat_teacher_direct_lethal_blend"

    def test_combat_teacher_runtime_override_bad_end_turn_only_when_baseline_ends_turn(self):
        from evaluate_ai import _combat_teacher_runtime_override_source

        state = _make_teacher_combat_state(
            [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
            enemy_hp=24,
        )
        legal = [
            {"action": "play_card", "card_index": 0, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=1,
            teacher_idx=0,
            runtime_labels={"bad_end_turn"},
        ) == "combat_teacher_rerank_bad_end_turn"
        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=0,
            teacher_idx=1,
            runtime_labels={"bad_end_turn"},
        ) is None

    def test_combat_teacher_runtime_override_ignores_potion_rerank_online(self):
        from evaluate_ai import _combat_teacher_runtime_override_source

        state = _make_teacher_combat_state(
            [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}],
            enemy_hp=18,
        )
        legal = [
            {"action": "use_potion", "index": 0, "label": "Fire Potion", "is_enabled": True},
            {"action": "play_card", "card_index": 0, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=0,
            teacher_idx=1,
            runtime_labels={"potion_misuse"},
        ) is None

    def test_combat_teacher_runtime_override_prefers_vulnerable_setup_over_plain_attack(self):
        from evaluate_ai import _combat_teacher_runtime_override_source

        state = _make_teacher_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=24,
        )
        legal = [
            {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=1,
            teacher_idx=0,
            runtime_labels={"bash_before_strike"},
        ) == "combat_teacher_rerank_bash_setup"
        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=0,
            teacher_idx=1,
            runtime_labels={"bash_before_strike"},
        ) is None

    def test_combat_teacher_runtime_override_prefers_block_before_body_slam(self):
        from evaluate_ai import _combat_teacher_runtime_override_source

        state = _make_teacher_combat_state(
            [
                {"id": "BODY_SLAM", "name": "Body Slam", "cost": 1, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=18,
            block=0,
        )
        legal = [
            {"action": "play_card", "card_index": 0, "label": "Body Slam", "card_id": "BODY_SLAM", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=0,
            teacher_idx=1,
            runtime_labels={"bodyslam_before_block"},
        ) == "combat_teacher_rerank_body_slam"
        assert _combat_teacher_runtime_override_source(
            state=state,
            legal=legal,
            baseline_idx=1,
            teacher_idx=0,
            runtime_labels={"bodyslam_before_block"},
        ) is None

    def test_combat_tactical_leaf_value_prefers_enemy_hp_progress(self):
        from evaluate_ai import _combat_tactical_leaf_value

        low_enemy_hp = {
            "state_type": "monster",
            "battle": {
                "player": {"hp": 60, "max_hp": 80, "block": 0},
                "enemies": [{"hp": 10, "max_hp": 40, "block": 0, "intents": [{"type": "attack", "damage": 6, "hits": 1}]}],
            },
        }
        high_enemy_hp = {
            "state_type": "monster",
            "battle": {
                "player": {"hp": 60, "max_hp": 80, "block": 0},
                "enemies": [{"hp": 35, "max_hp": 40, "block": 0, "intents": [{"type": "attack", "damage": 6, "hits": 1}]}],
            },
        }

        assert _combat_tactical_leaf_value(low_enemy_hp) > _combat_tactical_leaf_value(high_enemy_hp)

    def test_combat_tactical_leaf_value_rewards_useful_block(self):
        from evaluate_ai import _combat_tactical_leaf_value

        no_block = {
            "state_type": "monster",
            "battle": {
                "player": {"hp": 60, "max_hp": 80, "block": 0},
                "enemies": [{"hp": 25, "max_hp": 40, "block": 0, "intents": [{"type": "attack", "damage": 10, "hits": 1}]}],
            },
        }
        with_block = {
            "state_type": "monster",
            "battle": {
                "player": {"hp": 60, "max_hp": 80, "block": 8},
                "enemies": [{"hp": 25, "max_hp": 40, "block": 0, "intents": [{"type": "attack", "damage": 10, "hits": 1}]}],
            },
        }

        assert _combat_tactical_leaf_value(with_block) > _combat_tactical_leaf_value(no_block)

    def test_compute_summary_handles_all_error_results(self):
        from evaluate_ai import GameResult, compute_summary

        results = [
            GameResult(
                strategy="nn",
                game_id="g0",
                outcome="error",
                max_floor=0,
                final_hp=0,
                total_steps=0,
                num_combats_won=0,
                time_taken_s=0.0,
            )
        ]

        summary = compute_summary(results)

        assert summary["strategy"] == "nn"
        assert summary["total_games"] == 1
        assert summary["valid_games"] == 0
        assert summary["error_count"] == 1
        assert summary["act1_clear_rate"] == 0.0
        assert summary["combat_teacher_override_counts"] == {}

    def test_compute_summary_aggregates_teacher_override_counts(self):
        from evaluate_ai import GameResult, compute_summary

        results = [
            GameResult(
                strategy="nn",
                game_id="g0",
                outcome="death",
                max_floor=8,
                final_hp=0,
                total_steps=50,
                num_combats_won=3,
                time_taken_s=1.0,
                action_source_counts={
                    "nn": 10,
                    "combat_teacher_rerank_bad_end_turn": 2,
                },
                combat_teacher_override_counts={
                    "combat_teacher_rerank_bad_end_turn": 2,
                },
            ),
            GameResult(
                strategy="nn",
                game_id="g1",
                outcome="death",
                max_floor=10,
                final_hp=0,
                total_steps=60,
                num_combats_won=4,
                time_taken_s=2.0,
                action_source_counts={
                    "nn": 9,
                    "combat_teacher_rerank_bash_setup": 1,
                },
                combat_teacher_override_counts={
                    "combat_teacher_rerank_bash_setup": 1,
                },
            ),
        ]

        summary = compute_summary(results)

        assert summary["action_source_counts"]["nn"] == 19
        assert summary["combat_teacher_override_counts"]["combat_teacher_rerank_bad_end_turn"] == 2
        assert summary["combat_teacher_override_counts"]["combat_teacher_rerank_bash_setup"] == 1
        assert summary["avg_combat_teacher_overrides_per_game"] == pytest.approx(1.5)
        assert summary["games_with_combat_teacher_override"] == 2

    def test_write_trace_outputs_emits_index_and_trace(self, tmp_path):
        from evaluate_ai import GameResult, _write_trace_outputs

        results = [
            GameResult(
                game_id=1,
                seed="ironclad-benchmark-11",
                strategy="nn",
                max_floor=18,
                final_hp=0,
                final_max_hp=85,
                num_combats_won=10,
                total_steps=286,
                time_taken_s=3.8,
                outcome="death",
                boss_reached=True,
                act1_cleared=True,
                boss_hp_fraction_dealt=0.9722,
                floor_at_death=18,
            )
        ]
        traces = {
            "ironclad-benchmark-11": [
                {
                    "step": 0,
                    "state_type": "map",
                    "floor": 1,
                    "hp": 80,
                    "max_hp": 80,
                    "legal_action_count": 2,
                    "chosen_action": {"action": "choose_map_node", "index": 0},
                    "action_source": "nn",
                    "boss_hp_fraction_seen": 0.0,
                }
            ]
        }

        _write_trace_outputs(
            tmp_path,
            "nn",
            results,
            traces,
            {"checkpoint": "champion.pt", "trace_seeds": ["ironclad-benchmark-11"]},
        )

        index_path = tmp_path / "index.json"
        trace_path = tmp_path / "ironclad-benchmark-11_trace.json"

        assert index_path.exists()
        assert trace_path.exists()

        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))

        assert index_payload["captured_seeds"] == ["ironclad-benchmark-11"]
        assert index_payload["results"][0]["seed"] == "ironclad-benchmark-11"
        assert trace_payload["summary"]["act1_cleared"] is True
        assert trace_payload["trace"][0]["chosen_action"]["action"] == "choose_map_node"

    def test_partial_trace_selection_keeps_full_summary(self, tmp_path):
        from evaluate_ai import GameResult, _write_trace_outputs

        results = [
            GameResult(
                game_id=1,
                seed="ironclad-benchmark-07",
                strategy="nn",
                max_floor=16,
                final_hp=0,
                total_steps=200,
                num_combats_won=8,
                time_taken_s=2.0,
                outcome="death",
                boss_reached=True,
                boss_hp_fraction_dealt=0.7631,
                floor_at_death=16,
            ),
            GameResult(
                game_id=2,
                seed="ironclad-benchmark-11",
                strategy="nn",
                max_floor=18,
                final_hp=0,
                total_steps=286,
                num_combats_won=10,
                time_taken_s=3.8,
                outcome="death",
                boss_reached=True,
                act1_cleared=True,
                boss_hp_fraction_dealt=0.9722,
                floor_at_death=18,
            ),
        ]

        _write_trace_outputs(
            tmp_path,
            "nn",
            results,
            {
                "ironclad-benchmark-11": [
                    {
                        "step": 0,
                        "state_type": "boss",
                        "floor": 16,
                        "hp": 22,
                        "max_hp": 80,
                        "legal_action_count": 3,
                        "chosen_action": {"action": "play_card", "card_index": 0},
                        "action_source": "nn",
                        "boss_hp_fraction_seen": 0.8,
                    }
                ]
            },
            {"checkpoint": "champion.pt", "trace_seeds": ["ironclad-benchmark-11"]},
        )

        index_payload = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))

        assert len(index_payload["results"]) == 2
        assert {item["seed"] for item in index_payload["results"]} == {
            "ironclad-benchmark-07",
            "ironclad-benchmark-11",
        }
        assert (tmp_path / "ironclad-benchmark-11_trace.json").exists()
        assert not (tmp_path / "ironclad-benchmark-07_trace.json").exists()

    def test_write_trajectory_outputs_emits_manifest_and_jsonl(self, tmp_path):
        from evaluate_ai import GameResult, _write_trajectory_outputs

        results = [
            GameResult(
                game_id=1,
                seed="ironclad-benchmark-11",
                strategy="nn",
                max_floor=18,
                total_steps=1,
                outcome="death",
                boss_reached=True,
                act1_cleared=True,
                boss_hp_fraction_dealt=0.9722,
                floor_at_death=18,
            )
        ]
        trajectory_records = {
            "ironclad-benchmark-11": [
                {
                    "schema_version": "full_run_trajectory.v1",
                    "run_id": "run-1",
                    "step_index": 0,
                    "timestamp_utc": "2026-04-02T12:00:00Z",
                    "seed": "ironclad-benchmark-11",
                    "act": 1,
                    "floor": 16,
                    "state_type": "boss",
                    "raw_state": _make_v1_combat_state("boss", floor=16),
                    "candidate_actions": _make_v1_combat_state("boss", floor=16)["legal_actions"],
                    "chosen_action": {"action": "play_card", "card_index": 0, "target": "jaw-worm-0"},
                    "action_source": "combat_bc",
                    "next_state": {"state_type": "game_over", "terminal": True},
                    "terminal": True,
                    "run_outcome": "victory",
                    "delta": {
                        "state_changed": True,
                        "state_type_changed": True,
                        "changed_top_level_keys": ["battle", "state_type"],
                        "act_delta": 1,
                        "floor_delta": 1,
                        "hp_delta": 0,
                        "max_hp_delta": 0,
                        "gold_delta": 0,
                        "deck_count_delta": 0,
                        "relic_count_delta": 0,
                        "potion_count_delta": 0,
                    },
                }
            ]
        }

        _write_trajectory_outputs(
            tmp_path,
            "nn",
            results,
            trajectory_records,
            {"checkpoint": "champion.pt", "trajectory_seeds": ["ironclad-benchmark-11"]},
        )

        manifest_path = tmp_path / "trajectory_manifest.json"
        trajectory_path = tmp_path / "ironclad-benchmark-11_trajectory.jsonl"

        assert manifest_path.exists()
        assert trajectory_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["captured_seeds"] == ["ironclad-benchmark-11"]
        assert manifest["schema_version"] == "full_run_trajectory.v1"

        first_line = trajectory_path.read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(first_line)
        assert payload["chosen_action"]["action"] == "play_card"
        assert payload["terminal"] is True

    def test_trajectory_jsonl_round_trips_into_bc_exporter(self, tmp_path):
        from evaluate_ai import GameResult, _write_trajectory_outputs
        export_paths = _import_or_skip("export_combat_training_data_from_full_run").export_paths

        results = [
            GameResult(
                game_id=1,
                seed="ironclad-benchmark-19",
                strategy="nn",
                max_floor=16,
                total_steps=1,
                outcome="death",
                boss_reached=True,
                boss_hp_fraction_dealt=0.948,
                floor_at_death=16,
            )
        ]
        raw_state = _make_v1_combat_state("boss", floor=16)
        trajectory_records = {
            "ironclad-benchmark-19": [
                {
                    "schema_version": "full_run_trajectory.v1",
                    "run_id": "run-19",
                    "step_index": 0,
                    "timestamp_utc": "2026-04-02T12:00:00Z",
                    "seed": "ironclad-benchmark-19",
                    "act": 1,
                    "floor": 16,
                    "state_type": "boss",
                    "env_api_mode": "v1_singleplayer",
                    "raw_state": raw_state,
                    "candidate_actions": raw_state["legal_actions"],
                    "chosen_action": {"action": "play_card", "card_index": 0, "target": "jaw-worm-0"},
                    "action_source": "combat_bc",
                    "next_state": {"state_type": "game_over", "terminal": True},
                    "terminal": True,
                    "delta": {
                        "state_changed": True,
                        "state_type_changed": True,
                        "changed_top_level_keys": ["battle", "state_type"],
                        "act_delta": 0,
                        "floor_delta": 0,
                        "hp_delta": 0,
                        "max_hp_delta": 0,
                        "gold_delta": 0,
                        "deck_count_delta": 0,
                        "relic_count_delta": 0,
                        "potion_count_delta": 0,
                    },
                }
            ]
        }

        _write_trajectory_outputs(
            tmp_path / "trajectory",
            "nn",
            results,
            trajectory_records,
            {"checkpoint": "champion.pt"},
        )
        export_output = tmp_path / "bc_export.jsonl"
        summary = export_paths(
            [str(tmp_path / "trajectory" / "ironclad-benchmark-19_trajectory.jsonl")],
            export_output,
            allow_non_v1=False,
            min_floor=14,
            max_floor=17,
            pressure_incoming_damage=10,
            min_attacker_count=1,
            min_legal_actions=2,
            dedupe_state_action=True,
        )

        assert summary["written"] == 1
        assert export_output.exists()
        export_row = json.loads(export_output.read_text(encoding="utf-8").splitlines()[0])
        assert export_row["action"]["type"] == "play_card"
        assert export_row["info"]["floor"] == 16

    def test_build_boss_focus_dataset_exports_bucketed_rows(self, tmp_path):
        import sys
        build_boss_focus_main = _import_or_skip("build_boss_focus_dataset").main
        from evaluate_ai import GameResult, _write_trajectory_outputs

        results = [
            GameResult(
                game_id=1,
                seed="ironclad-benchmark-11",
                strategy="nn",
                max_floor=18,
                total_steps=1,
                outcome="death",
                boss_reached=True,
                act1_cleared=True,
                boss_hp_fraction_dealt=0.97,
                floor_at_death=18,
            ),
            GameResult(
                game_id=2,
                seed="ironclad-benchmark-19",
                strategy="nn",
                max_floor=16,
                total_steps=1,
                outcome="death",
                boss_reached=True,
                act1_cleared=False,
                boss_hp_fraction_dealt=0.94,
                floor_at_death=16,
            ),
        ]
        boss_state = _make_v1_combat_state("boss", floor=16)
        trajectory_dir = tmp_path / "trajectory"
        _write_trajectory_outputs(
            trajectory_dir,
            "nn",
            results,
            {
                "ironclad-benchmark-11": [
                    {
                        "schema_version": "full_run_trajectory.v1",
                        "run_id": "run-11",
                        "step_index": 0,
                        "timestamp_utc": "2026-04-02T12:00:00Z",
                        "seed": "ironclad-benchmark-11",
                        "act": 1,
                        "floor": 16,
                        "state_type": "boss",
                        "env_api_mode": "v1_singleplayer",
                        "raw_state": boss_state,
                        "candidate_actions": boss_state["legal_actions"],
                        "chosen_action": {"action": "play_card", "card_index": 0, "target": "jaw-worm-0"},
                        "action_source": "nn",
                        "next_state": boss_state,
                        "terminal": False,
                        "delta": {"hp_delta": 0},
                    }
                ],
                "ironclad-benchmark-19": [
                    {
                        "schema_version": "full_run_trajectory.v1",
                        "run_id": "run-19",
                        "step_index": 0,
                        "timestamp_utc": "2026-04-02T12:00:00Z",
                        "seed": "ironclad-benchmark-19",
                        "act": 1,
                        "floor": 16,
                        "state_type": "boss",
                        "env_api_mode": "v1_singleplayer",
                        "raw_state": boss_state,
                        "candidate_actions": boss_state["legal_actions"],
                        "chosen_action": {"action": "play_card", "card_index": 0, "target": "jaw-worm-0"},
                        "action_source": "nn",
                        "next_state": boss_state,
                        "terminal": False,
                        "delta": {"hp_delta": 0},
                    }
                ],
            },
            {"checkpoint": "champion.pt"},
        )

        clear_path = tmp_path / "clear.json"
        target_path = tmp_path / "target.json"
        clear_path.write_text(json.dumps(["ironclad-benchmark-11"]), encoding="utf-8")
        target_path.write_text(json.dumps(["ironclad-benchmark-19"]), encoding="utf-8")
        output_path = tmp_path / "boss_focus.jsonl"
        manifest_path = tmp_path / "boss_focus_manifest.json"

        argv_backup = sys.argv[:]
        try:
            sys.argv = [
                "build_boss_focus_dataset.py",
                str(trajectory_dir / "ironclad-benchmark-11_trajectory.jsonl"),
                str(trajectory_dir / "ironclad-benchmark-19_trajectory.jsonl"),
                "--output",
                str(output_path),
                "--manifest-output",
                str(manifest_path),
                "--clear-seeds-file",
                str(clear_path),
                "--target-seeds-file",
                str(target_path),
                "--clear-repeat",
                "2",
                "--target-repeat",
                "1",
            ]
            assert build_boss_focus_main() == 0
        finally:
            sys.argv = argv_backup

        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert len(rows) == 3
        assert rows[0]["focus_bucket"] == "clear_reference"
        assert rows[-1]["focus_bucket"] == "near_win_target"
        assert manifest["bucket_counts"]["clear_reference"] == 1
        assert manifest["bucket_counts"]["near_win_target"] == 1
        assert manifest["written_rows_after_repeat"] == 3

    def test_build_boss_focus_dataset_respects_enemy_and_round_filters(self, tmp_path):
        import sys
        build_boss_focus_main = _import_or_skip("build_boss_focus_dataset").main
        from evaluate_ai import GameResult, _write_trajectory_outputs

        jaw_boss_state = _make_v1_combat_state("boss", floor=16)
        cult_boss_state = _make_v1_combat_state("boss", floor=16)
        cult_boss_state["battle"]["round"] = 1
        cult_boss_state["battle"]["enemies"] = [
            {
                "entity_id": "cult-a",
                "combat_id": 1,
                "name": "Cultist A",
                "hp": 20,
                "max_hp": 20,
                "block": 0,
                "status": [],
                "intents": [{"type": "Attack", "label": "5", "title": "Attack", "description": "Deal 5."}],
            },
            {
                "entity_id": "cult-b",
                "combat_id": 2,
                "name": "Cultist B",
                "hp": 20,
                "max_hp": 20,
                "block": 0,
                "status": [],
                "intents": [{"type": "Attack", "label": "5", "title": "Attack", "description": "Deal 5."}],
            },
        ]
        cult_boss_state["legal_actions"] = [
            {"action": "play_card", "card_index": 0, "target_id": 1, "label": "Strike A"},
            {"action": "play_card", "card_index": 0, "target_id": 2, "label": "Strike B"},
            {"action": "end_turn", "label": "End Turn"},
        ]

        results = [
            GameResult(
                game_id=1,
                seed="ironclad-benchmark-11",
                strategy="nn",
                max_floor=18,
                total_steps=1,
                outcome="death",
                boss_reached=True,
                act1_cleared=True,
                boss_hp_fraction_dealt=0.97,
                floor_at_death=18,
            ),
            GameResult(
                game_id=2,
                seed="ironclad-benchmark-19",
                strategy="nn",
                max_floor=16,
                total_steps=1,
                outcome="death",
                boss_reached=True,
                act1_cleared=False,
                boss_hp_fraction_dealt=0.94,
                floor_at_death=16,
            ),
        ]
        trajectory_dir = tmp_path / "trajectory"
        _write_trajectory_outputs(
            trajectory_dir,
            "nn",
            results,
            {
                "ironclad-benchmark-11": [
                    {
                        "schema_version": "full_run_trajectory.v1",
                        "run_id": "run-11",
                        "step_index": 0,
                        "timestamp_utc": "2026-04-02T12:00:00Z",
                        "seed": "ironclad-benchmark-11",
                        "act": 1,
                        "floor": 16,
                        "state_type": "boss",
                        "env_api_mode": "v1_singleplayer",
                        "raw_state": jaw_boss_state,
                        "candidate_actions": jaw_boss_state["legal_actions"],
                        "chosen_action": {"action": "play_card", "card_index": 0, "target": "jaw-worm-0"},
                        "action_source": "nn",
                        "next_state": jaw_boss_state,
                        "terminal": False,
                        "delta": {"hp_delta": 0},
                    }
                ],
                "ironclad-benchmark-19": [
                    {
                        "schema_version": "full_run_trajectory.v1",
                        "run_id": "run-19",
                        "step_index": 0,
                        "timestamp_utc": "2026-04-02T12:00:00Z",
                        "seed": "ironclad-benchmark-19",
                        "act": 1,
                        "floor": 16,
                        "state_type": "boss",
                        "env_api_mode": "v1_singleplayer",
                        "raw_state": cult_boss_state,
                        "candidate_actions": cult_boss_state["legal_actions"],
                        "chosen_action": {"action": "play_card", "card_index": 0, "target": "cult-a"},
                        "action_source": "nn",
                        "next_state": cult_boss_state,
                        "terminal": False,
                        "delta": {"hp_delta": 0},
                    }
                ],
            },
            {"checkpoint": "champion.pt"},
        )

        clear_path = tmp_path / "clear.json"
        target_path = tmp_path / "target.json"
        clear_path.write_text(json.dumps(["ironclad-benchmark-11"]), encoding="utf-8")
        target_path.write_text(json.dumps(["ironclad-benchmark-19"]), encoding="utf-8")
        output_path = tmp_path / "boss_focus_filtered.jsonl"
        manifest_path = tmp_path / "boss_focus_filtered_manifest.json"

        argv_backup = sys.argv[:]
        try:
            sys.argv = [
                "build_boss_focus_dataset.py",
                str(trajectory_dir / "ironclad-benchmark-11_trajectory.jsonl"),
                str(trajectory_dir / "ironclad-benchmark-19_trajectory.jsonl"),
                "--output",
                str(output_path),
                "--manifest-output",
                str(manifest_path),
                "--clear-seeds-file",
                str(clear_path),
                "--target-seeds-file",
                str(target_path),
                "--enemy-name-token",
                "jaw",
                "--max-alive-enemies",
                "1",
                "--min-round",
                "1",
                "--max-round",
                "2",
            ]
            assert build_boss_focus_main() == 0
        finally:
            sys.argv = argv_backup

        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert len(rows) == 3
        assert {row["seed"] for row in rows} == {"ironclad-benchmark-11"}
        assert manifest["skipped_enemy_name_filter"] >= 1
        assert manifest["filters"]["enemy_name_tokens"] == ["jaw"]
        assert manifest["filters"]["max_alive_enemies"] == 1

    def test_combat_bc_rerank_selects_action_and_honors_gate(self, tmp_path):
        from archive.combat_bc import FEATURE_NAMES
        from evaluate_ai import (
            _combat_bc_gate_allows_state,
            _load_combat_bc_override,
            _select_action_combat_bc,
        )

        model_path = tmp_path / "combat_behavior_clone.json"
        model_payload = {
            "model_type": "linear_softmax_state_action",
            "feature_names": FEATURE_NAMES,
            "weights": [0.0] * len(FEATURE_NAMES),
            "gate_constraints": {
                "min_floor": 14,
                "room_types": ["elite", "boss"],
                "enemy_name_tokens_any": ["jaw"],
                "max_alive_enemies": 1,
            },
            "patch_config": {
                "mode": "rerank",
                "alpha": 0.5,
                "require_margin": False,
                "min_margin_zscore": 0.0,
            },
        }
        model_path.write_text(json.dumps(model_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        override = _load_combat_bc_override(model_path)
        boss_state = _make_v1_combat_state("boss", floor=16)
        early_state = _make_v1_combat_state("monster", floor=10)
        wrong_boss_state = _make_v1_combat_state("boss", floor=16)
        wrong_boss_state["battle"]["enemies"][0]["name"] = "Hexaghost"
        base_logits = np.zeros((len(boss_state["legal_actions"]),), dtype=np.float64)

        assert _combat_bc_gate_allows_state(boss_state, override.gate_constraints) is True
        assert _combat_bc_gate_allows_state(early_state, override.gate_constraints) is False
        assert _combat_bc_gate_allows_state(wrong_boss_state, override.gate_constraints) is False

        action_idx, action, action_source = _select_action_combat_bc(
            state=boss_state,
            legal=boss_state["legal_actions"],
            combat_bc_override=override,
            base_logits=base_logits,
        )
        assert action_idx == 0
        assert action["action"] == "play_card"
        assert action_source == "combat_bc_rerank"

    def test_combat_bc_rerank_margin_guard_falls_back(self, tmp_path):
        from archive.combat_bc import FEATURE_NAMES
        from evaluate_ai import _load_combat_bc_override, _select_action_combat_bc

        model_path = tmp_path / "combat_behavior_clone.json"
        model_payload = {
            "model_type": "linear_softmax_state_action",
            "feature_names": FEATURE_NAMES,
            "weights": [0.0] * len(FEATURE_NAMES),
            "gate_constraints": {"min_floor": 14, "room_types": ["elite", "boss"]},
            "patch_config": {
                "mode": "rerank",
                "alpha": 0.5,
                "require_margin": True,
                "min_margin_zscore": 0.75,
            },
        }
        model_path.write_text(json.dumps(model_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        override = _load_combat_bc_override(model_path)
        boss_state = _make_v1_combat_state("boss", floor=16)
        base_logits = np.asarray([3.0] + [0.0] * (len(boss_state["legal_actions"]) - 1), dtype=np.float64)

        choice = _select_action_combat_bc(
            state=boss_state,
            legal=boss_state["legal_actions"],
            combat_bc_override=override,
            base_logits=base_logits,
        )

        assert choice is None

    def test_combat_bc_rerank_respects_base_confidence_gate(self, tmp_path):
        from archive.combat_bc import FEATURE_NAMES
        from evaluate_ai import _load_combat_bc_override, _select_action_combat_bc

        model_path = tmp_path / "combat_behavior_clone.json"
        model_payload = {
            "model_type": "linear_softmax_state_action",
            "feature_names": FEATURE_NAMES,
            "weights": [0.0] * len(FEATURE_NAMES),
            "gate_constraints": {"min_floor": 14, "room_types": ["boss"]},
            "patch_config": {
                "mode": "rerank",
                "alpha": 0.5,
                "require_margin": False,
                "min_margin_zscore": 0.0,
                "max_base_top_prob": 0.7,
            },
        }
        model_path.write_text(json.dumps(model_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        override = _load_combat_bc_override(model_path)
        boss_state = _make_v1_combat_state("boss", floor=16)
        base_logits = np.asarray([4.0] + [0.0] * (len(boss_state["legal_actions"]) - 1), dtype=np.float64)

        choice = _select_action_combat_bc(
            state=boss_state,
            legal=boss_state["legal_actions"],
            combat_bc_override=override,
            base_logits=base_logits,
        )

        assert choice is None


class TestStage25PlanAssets:
    def test_stage25_manifest_and_seed_buckets_exist(self):
        repo_root = Path(__file__).resolve().parents[2]
        manifest_path = repo_root / "artifacts" / "combat_training" / "seed_sets" / "act1_stage25_plan_manifest.json"
        clear_path = repo_root / "artifacts" / "combat_training" / "seed_sets" / "act1_stage25_clear_reference_seeds.json"
        near_win_path = repo_root / "artifacts" / "combat_training" / "seed_sets" / "act1_stage25_near_win_boss_specialist_seeds.json"
        boss_diag_path = repo_root / "artifacts" / "combat_training" / "seed_sets" / "act1_stage25_boss_no_answer_diagnostics.json"
        early_gate_path = repo_root / "artifacts" / "combat_training" / "seed_sets" / "act1_stage25_early_regression_gate_seeds.json"

        required_paths = [
            manifest_path,
            clear_path,
            near_win_path,
            boss_diag_path,
            early_gate_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            pytest.skip(f"stage25 seed-set artifacts not present in this workspace: {missing}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clear_seeds = json.loads(clear_path.read_text(encoding="utf-8"))
        near_win_seeds = json.loads(near_win_path.read_text(encoding="utf-8"))
        boss_diag_seeds = json.loads(boss_diag_path.read_text(encoding="utf-8"))
        early_gate_seeds = json.loads(early_gate_path.read_text(encoding="utf-8"))

        assert manifest["stage"] == "act1_stage_2_5"
        assert manifest["baseline"]["summary"]["act1_clear_rate"] == 0.15
        assert manifest["baseline"]["summary"]["act1_clear_count"] == 3
        assert manifest["baseline"]["summary"]["boss_reach_rate"] == 0.55
        assert manifest["baseline"]["summary"]["boss_reach_count"] == 11
        assert manifest["baseline"]["summary"]["avg_boss_hp_fraction_dealt"] == 0.6152
        assert manifest["baseline"]["summary"]["near_win_conversion_count"] == 0
        assert manifest["promotion_gate"]["min_avg_floor"] == 12.5
        assert manifest["promotion_gate"]["max_early_death_count"] == 5
        assert manifest["promotion_gate"]["safety_gate"]["max_error_count"] == 0
        assert manifest["promotion_gate"]["safety_gate"]["max_timeout_count"] == 0
        assert manifest["promotion_gate"]["lexicographic_order"] == [
            "act1_clear_count",
            "near_win_conversion_count",
            "boss_reach_count",
            "avg_boss_hp_fraction_dealt",
            "avg_floor",
        ]
        assert manifest["benchmark_eval"]["trace_focus_seeds"] == [
            "ironclad-benchmark-07",
            "ironclad-benchmark-11",
            "ironclad-benchmark-19",
        ]
        assert clear_seeds == [
            "ironclad-benchmark-08",
            "ironclad-benchmark-11",
            "ironclad-benchmark-12",
        ]
        assert near_win_seeds == [
            "ironclad-benchmark-07",
            "ironclad-benchmark-14",
            "ironclad-benchmark-19",
        ]
        assert boss_diag_seeds == [
            "ironclad-benchmark-02",
            "ironclad-benchmark-03",
            "ironclad-benchmark-09",
        ]
        assert early_gate_seeds == [
            "ironclad-benchmark-06",
            "ironclad-benchmark-10",
            "ironclad-benchmark-13",
            "ironclad-benchmark-15",
            "ironclad-benchmark-20",
        ]

class TestBossAwarePlanning:
    def test_extract_next_boss_token_and_structured_state(self, vocab):
        from rl_reward_shaping import extract_next_boss_token, boss_readiness_score
        from rl_encoder_v2 import build_structured_state
        from rl_policy_v2 import _structured_state_to_numpy_dict

        state = _make_synthetic_state("map")
        assert extract_next_boss_token(state) == "ceremonial_beast"
        readiness = boss_readiness_score(state)
        assert 0.0 <= readiness <= 1.0

        ss = build_structured_state(state, vocab)
        flat = _structured_state_to_numpy_dict(ss)
        assert ss.next_boss_idx > 0
        assert int(flat["next_boss_idx"]) == ss.next_boss_idx

    def test_structured_rollout_buffer_keeps_boss_readiness_targets(self, vocab):
        from rl_encoder_v2 import build_structured_state, build_structured_actions
        from rl_policy_v2 import StructuredRolloutBuffer

        state = _make_synthetic_state("map")
        actions = _make_synthetic_actions("map")
        ss = build_structured_state(state, vocab)
        sa = build_structured_actions(state, actions, vocab)

        buf = StructuredRolloutBuffer()
        buf.add(ss, sa, action_idx=0, log_prob=-0.2, reward=0.1, value=0.3, done=False, boss_readiness_target=0.62)
        buf.compute_gae()
        tensors = buf.to_tensors()
        assert "boss_readiness_targets" in tensors
        assert tensors["boss_readiness_targets"].shape == (1,)
        assert tensors["boss_readiness_targets"][0].item() == pytest.approx(0.62)

    def test_boss_aware_warmup_freezes_old_ppo_params(self, vocab):
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from train_hybrid import _configure_boss_aware_warmup

        net = FullRunPolicyNetworkV2(vocab=vocab)
        trainable, total = _configure_boss_aware_warmup(net)
        assert 0 < trainable < total
        assert net.boss_screen_adapter.weight.requires_grad is True
        assert net.entity_emb.text_token_embed.weight.requires_grad is True
        assert net.trunk.net[0].weight.requires_grad is False

    def test_compute_summary_includes_boss_readiness_and_skip_rates(self):
        from evaluate_ai import GameResult, compute_summary

        results = [
            GameResult(
                game_id=1,
                seed="s1",
                strategy="nn",
                max_floor=18,
                outcome="death",
                boss_reached=True,
                act1_cleared=True,
                boss_hp_fraction_dealt=0.95,
                boss_readiness_at_floor_8=0.42,
                boss_readiness_at_floor_12=0.51,
                boss_readiness_at_floor_16=0.66,
                next_boss_token="ceremonial_beast",
                card_reward_screens=2,
                card_reward_skips=1,
                total_steps=100,
                time_taken_s=1.0,
            ),
            GameResult(
                game_id=2,
                seed="s2",
                strategy="nn",
                max_floor=16,
                outcome="death",
                boss_reached=True,
                act1_cleared=False,
                boss_hp_fraction_dealt=0.50,
                boss_readiness_at_floor_8=0.30,
                boss_readiness_at_floor_12=0.45,
                boss_readiness_at_floor_16=0.55,
                next_boss_token="ceremonial_beast",
                card_reward_screens=1,
                card_reward_skips=0,
                total_steps=80,
                time_taken_s=1.0,
            ),
        ]
        summary = compute_summary(results)
        assert summary["avg_boss_readiness_at_floor_8"] == pytest.approx(0.36, abs=1e-4)
        assert summary["avg_boss_readiness_at_floor_12"] == pytest.approx(0.48, abs=1e-4)
        assert summary["avg_boss_readiness_at_floor_16"] == pytest.approx(0.605, abs=1e-4)
        assert summary["card_reward_skip_rate_by_boss"]["ceremonial_beast"] == pytest.approx(1 / 3, abs=1e-4)


class TestNnBackendParityAudit:
    def test_tensor_diff_summary_tracks_float_and_bool_differences(self):
        _tensor_diff_summary = _import_or_skip("nn_backend_parity_audit")._tensor_diff_summary

        float_left = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        float_right = torch.tensor([[0.0, 1.25]], dtype=torch.float32)
        bool_left = torch.tensor([True, False, True], dtype=torch.bool)
        bool_right = torch.tensor([True, True, True], dtype=torch.bool)

        float_summary = _tensor_diff_summary(float_left, float_right)
        bool_summary = _tensor_diff_summary(bool_left, bool_right)

        assert float_summary["shape_match"] is True
        assert float_summary["allclose"] is False
        assert float_summary["max_abs_diff"] == pytest.approx(0.25)
        assert bool_summary["dtype"] == "bool"
        assert bool_summary["mismatch_count"] == 1

    def test_find_matching_action_index_uses_normalized_legal_action_key(self):
        _find_matching_action_index = _import_or_skip("nn_backend_parity_audit")._find_matching_action_index

        driver = {"action": "play_card", "card_index": 2, "target_id": "enemy-a", "is_enabled": True}
        legal = [
            {"action": "end_turn", "is_enabled": True},
            {"action": "play_card", "card_index": 2, "target_id": "enemy-a", "is_enabled": True},
        ]

        assert _find_matching_action_index(driver, legal) == 1

    def test_compare_logits_detects_argmax_match_and_diff_scale(self):
        _compare_logits = _import_or_skip("nn_backend_parity_audit")._compare_logits

        left = np.asarray([0.1, 0.8, 0.2], dtype=np.float64)
        right = np.asarray([0.0, 0.7, 0.1], dtype=np.float64)
        summary = _compare_logits(left, right)

        assert summary["shape_match"] is True
        assert summary["argmax_match"] is True
        assert summary["max_abs_diff"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Combat-First Teacher Stack V1
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Combat teacher/microbench dataset stack is no longer part of the STS2AI mainline.")
class TestCombatTeacherStackV1:
    def test_forward_teacher_shapes(self, vocab):
        from combat_nn import CombatPolicyValueNetwork, build_combat_action_features, build_combat_features

        state = _make_teacher_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ]
        )
        state["legal_actions"] = [
            {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        net = CombatPolicyValueNetwork(vocab=vocab)
        sf = build_combat_features(state, vocab)
        af = build_combat_action_features(state, state["legal_actions"], vocab)

        state_t = {}
        for key, value in sf.items():
            tensor = torch.tensor(value).unsqueeze(0)
            if value.dtype in (np.int64, np.int32):
                tensor = tensor.long()
            elif value.dtype == bool:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()
            state_t[key] = tensor
        action_t = {}
        for key, value in af.items():
            tensor = torch.tensor(value).unsqueeze(0)
            if value.dtype in (np.int64, np.int32):
                tensor = tensor.long()
            elif value.dtype == bool:
                tensor = tensor.bool()
            else:
                tensor = tensor.float()
            action_t[key] = tensor

        with torch.no_grad():
            logits, value, action_scores, continuation = net.forward_teacher(state_t, action_t)

        assert logits.shape == action_scores.shape
        assert continuation.shape == (1, 3)
        assert float(continuation[0, 0].item()) >= 0.0
        assert float(continuation[0, 1].item()) >= 0.0
        assert float(continuation[0, 2].item()) >= 0.0

    def test_canonical_public_state_hash_ignores_hidden_draw_order(self):
        from combat_teacher_common import canonical_public_state_hash

        base = _make_teacher_combat_state(
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ]
        )
        base["legal_actions"] = [{"action": "end_turn", "is_enabled": True}]
        a = copy.deepcopy(base)
        b = copy.deepcopy(base)
        a["battle"]["player"]["draw_pile"] = [{"id": "A"}, {"id": "B"}]
        b["battle"]["player"]["draw_pile"] = [{"id": "B"}, {"id": "A"}]

        assert canonical_public_state_hash(a) == canonical_public_state_hash(b)

        b["battle"]["player"]["block"] = 4
        assert canonical_public_state_hash(a) != canonical_public_state_hash(b)

    def test_turn_solver_prefers_bash_before_strike(self, vocab):
        from combat_turn_solver import CombatTurnSolver

        state = _make_teacher_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=20,
        )
        state["legal_actions"] = [
            {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        env = _FakeCombatTurnEnv(state)
        solver = CombatTurnSolver(env, _DummyBaselinePolicy())
        solution = solver.solve(state, root_state_id=env.save_state())

        assert solution.supported is True
        assert solution.best_first_action is not None
        assert solution.best_first_action["card_id"] == "BASH"
        assert solution.best_full_turn_line[0]["card_id"] == "BASH"

    def test_transposition_cache_hits_on_repeated_solve(self, vocab):
        from combat_turn_solver import CombatTurnSolver

        state = _make_teacher_combat_state(
            [
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                {"id": "BODY_SLAM", "name": "Body Slam", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=16,
        )
        state["legal_actions"] = [
            {"action": "play_card", "card_index": 0, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Body Slam", "card_id": "BODY_SLAM", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        env = _FakeCombatTurnEnv(state)
        solver = CombatTurnSolver(env, _DummyBaselinePolicy())
        first = solver.solve(state, root_state_id=env.save_state())
        second = solver.solve(state, root_state_id=env.save_state())

        assert first.supported is True
        assert second.supported is True
        assert second.search_stats["cache_hits"] > 0

    def test_teacher_dataset_roundtrip_and_microbench(self, tmp_path):
        from combat_microbench import build_microbench_report
        from combat_teacher_dataset import CombatTeacherSample, load_combat_teacher_samples, write_combat_teacher_samples

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-aa",
            split="holdout",
            source_bucket="motif",
            source_seed="S1",
            source_checkpoint="combat.pt",
            state_hash="hash-aa",
            motif_labels=["bash_before_strike"],
            state=_make_teacher_combat_state(
                [
                    {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                    {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                ],
                enemy_hp=20,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
                {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2, 0.0],
            baseline_probs=[0.34, 0.38, 0.28],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "BASH"}],
            per_action_score=[1.0, 0.5, -0.2],
            per_action_regret=[0.0, 0.5, 1.2],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 0.7, "expected_hp_loss": 4.0, "expected_potion_cost": 0.0},
        )
        path = tmp_path / "teacher.jsonl"
        write_combat_teacher_samples(path, [sample], metadata={"unit_test": True})
        loaded = load_combat_teacher_samples(path)
        assert len(loaded) == 1
        report = build_microbench_report(loaded)
        assert report["schema_version"] == "combat_microbench.v1"
        assert report["sample_count"] == 1
        assert report["source_sample_count"] == 1

    def test_microbench_dedupes_duplicate_sample_ids(self):
        from combat_microbench import build_microbench_report
        from combat_teacher_dataset import CombatTeacherSample

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-dup",
            split="holdout",
            source_bucket="motif",
            source_seed="SD",
            source_checkpoint="combat.pt",
            state_hash="hash-dup",
            motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
            state=_make_teacher_combat_state(
                [
                    {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                ],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.3],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, -0.5],
            per_action_regret=[0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )
        report = build_microbench_report([sample, sample, sample])
        assert report["source_sample_count"] == 3
        assert report["sample_count"] == 1

    def test_strict_bash_metric_excludes_tied_solver_states(self):
        from combat_teacher_dataset import CombatTeacherSample, sample_metric_applicable

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-bash-tie",
            split="holdout",
            source_bucket="motif",
            source_seed="ST",
            source_checkpoint="combat.pt",
            state_hash="hash-bash-tie",
            motif_labels=["bash_before_strike", "bad_end_turn"],
            state=_make_teacher_combat_state(
                [
                    {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                    {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                    {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                ],
                enemy_hp=20,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
                {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "play_card", "card_index": 2, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.0, 0.0, 0.0, 0.0],
            baseline_probs=[0.25, 0.25, 0.25, 0.25],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "BASH"}],
            per_action_score=[1.0, 0.8, 1.0, -0.2],
            per_action_regret=[0.0, 0.2, 0.0, 1.2],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 0.7, "expected_hp_loss": 4.0, "expected_potion_cost": 0.0},
        )
        assert sample_metric_applicable(sample, "bash_before_strike") is False

    def test_microbench_uses_regret_not_exact_index_for_errors(self):
        from combat_microbench import evaluate_policy_on_samples
        from combat_teacher_dataset import CombatTeacherSample

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-zero-regret-alt",
            split="holdout",
            source_bucket="motif",
            source_seed="SZ",
            source_checkpoint="combat.pt",
            state_hash="hash-zero-regret-alt",
            motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
            state=_make_teacher_combat_state(
                [
                    {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                    {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                ],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "play_card", "card_index": 1, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2, 0.0],
            baseline_probs=[0.3, 0.6, 0.1],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, 1.0, -0.5],
            per_action_regret=[0.0, 0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )

        class _AltZeroRegretPolicy:
            name = "alt-zero-regret"

            def choose_action_index(self, _sample):
                return 1

        report = evaluate_policy_on_samples([sample], _AltZeroRegretPolicy())
        assert report["metrics"]["missed_lethal_rate"] == 0.0
        assert report["metrics"]["direct_lethal_first_action_error_rate"] == 0.0

    def test_teacher_microbench_selective_lethal_blend_prefers_logit_supported_lethal(self, vocab):
        import torch

        from combat_microbench import TeacherSamplePolicy
        from combat_teacher_dataset import CombatTeacherSample

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="teacher-lethal-blend",
            split="holdout",
            source_bucket="on_policy",
            source_seed="TLB",
            source_checkpoint="combat.pt",
            state_hash="hash-teacher-lethal-blend",
            motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
            state=_make_teacher_combat_state(
                [
                    {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                    {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                ],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "play_card", "card_index": 1, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.2, 0.7, 0.1],
            baseline_probs=[0.2, 0.7, 0.1],
            baseline_best_action_index=1,
            best_action_index=1,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[0.5, 1.0, -0.2],
            per_action_regret=[0.5, 0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )

        class _FakeTeacherNet:
            def forward_teacher(self, _state_t, _action_t):
                logits = torch.tensor([[0.1, 3.4, -1.0]], dtype=torch.float32)
                value = torch.tensor([0.0], dtype=torch.float32)
                scores = torch.tensor([[2.0, 0.5, 1.0]], dtype=torch.float32)
                continuation = torch.zeros((1, 3), dtype=torch.float32)
                return logits, value, scores, continuation

        plain_policy = TeacherSamplePolicy(
            network=_FakeTeacherNet(),
            vocab=vocab,
            device=torch.device("cpu"),
            lethal_logit_blend_alpha=0.0,
        )
        blended_policy = TeacherSamplePolicy(
            network=_FakeTeacherNet(),
            vocab=vocab,
            device=torch.device("cpu"),
            lethal_logit_blend_alpha=0.5,
        )

        assert plain_policy.choose_action_index(sample) == 0
        assert blended_policy.choose_action_index(sample) == 1

    def test_teacher_trainer_tiny_overfit(self, vocab):
        from torch.utils.data import DataLoader

        from combat_teacher_dataset import CombatTeacherSample
        from train_combat_teacher import CombatTeacherTorchDataset, _run_epoch
        from combat_nn import CombatPolicyValueNetwork

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-bb",
            split="train",
            source_bucket="motif",
            source_seed="S2",
            source_checkpoint="combat.pt",
            state_hash="hash-bb",
            motif_labels=["bodyslam_before_block"],
            state=_make_teacher_combat_state(
                [
                    {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
                    {"id": "BODY_SLAM", "name": "Body Slam", "cost": 1, "is_upgraded": False},
                ],
                enemy_hp=18,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "play_card", "card_index": 1, "label": "Body Slam", "card_id": "BODY_SLAM", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.2, 0.5, 0.1],
            baseline_probs=[0.32, 0.48, 0.20],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[1.0, 0.1, -0.2],
            per_action_regret=[0.0, 0.9, 1.2],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 0.8, "expected_hp_loss": 3.0, "expected_potion_cost": 0.0},
        )
        dataset = CombatTeacherTorchDataset([sample, sample], vocab=vocab)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=list)
        net = CombatPolicyValueNetwork(vocab=vocab)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
        before = _run_epoch(net, loader, optimizer=None, device=torch.device("cpu"))
        for _ in range(25):
            _run_epoch(net, loader, optimizer=optimizer, device=torch.device("cpu"))
        after = _run_epoch(net, loader, optimizer=None, device=torch.device("cpu"))

        assert after.loss < before.loss
        assert after.teacher_best_action_ce < before.teacher_best_action_ce

    def test_dataset_assembly_preserves_scarce_motifs(self):
        from build_combat_teacher_dataset import _assemble_dataset
        from combat_teacher_dataset import CombatTeacherSample

        bash_state = _make_teacher_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=20,
        )
        bash_actions = [
            {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        bash_sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-bash",
            split="train",
            source_bucket="motif",
            source_seed="SB",
            source_checkpoint="combat.pt",
            state_hash="hash-bash",
            motif_labels=["bash_before_strike", "bad_end_turn"],
            state=bash_state,
            legal_actions=bash_actions,
            baseline_logits=[0.1, 0.2, 0.0],
            baseline_probs=[0.34, 0.38, 0.28],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "BASH"}],
            per_action_score=[1.0, 0.5, -0.2],
            per_action_regret=[0.0, 0.5, 1.2],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 0.7, "expected_hp_loss": 4.0, "expected_potion_cost": 0.0},
        )

        lethal_state = _make_teacher_combat_state(
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=6,
        )
        lethal_actions = [
            {"action": "play_card", "card_index": 0, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        lethal_sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-lethal",
            split="train",
            source_bucket="motif",
            source_seed="SL",
            source_checkpoint="combat.pt",
            state_hash="hash-lethal",
            motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
            state=lethal_state,
            legal_actions=lethal_actions,
            baseline_logits=[0.1, 0.3],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, -0.5],
            per_action_regret=[0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )
        turn_lethal_state = _make_teacher_combat_state(
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=12,
        )
        turn_lethal_actions = [
            {"action": "play_card", "card_index": 0, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 2, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        turn_lethal_sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-turn-lethal",
            split="train",
            source_bucket="motif",
            source_seed="STL",
            source_checkpoint="combat.pt",
            state_hash="hash-turn-lethal",
            motif_labels=["missed_lethal", "turn_lethal_no_end_turn", "bad_end_turn"],
            state=turn_lethal_state,
            legal_actions=turn_lethal_actions,
            baseline_logits=[0.1, 0.1, 0.4, 0.0],
            baseline_probs=[0.2, 0.2, 0.5, 0.1],
            baseline_best_action_index=2,
            best_action_index=0,
            best_full_turn_line=[
                {"action": "play_card", "card_id": "STRIKE_IRONCLAD"},
                {"action": "play_card", "card_id": "STRIKE_IRONCLAD"},
            ],
            per_action_score=[1.0, 1.0, 0.2, -0.8],
            per_action_regret=[0.0, 0.0, 0.8, 1.8],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )

        bad_end_state = _make_teacher_combat_state(
            [
                {"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=30,
        )
        bad_end_actions = [
            {"action": "play_card", "card_index": 0, "label": "Defend", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        bad_end_sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sample-bad-end",
            split="train",
            source_bucket="on_policy",
            source_seed="SE",
            source_checkpoint="combat.pt",
            state_hash="hash-end",
            motif_labels=["bad_end_turn"],
            state=bad_end_state,
            legal_actions=bad_end_actions,
            baseline_logits=[0.1, 0.3],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[0.5, -0.2],
            per_action_regret=[0.0, 0.7],
            root_value=0.5,
            leaf_breakdown={"total": 0.5},
            continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
        )

        on_policy_pool = []
        for idx in range(20):
            sample_copy = copy.deepcopy(bad_end_sample)
            sample_copy.sample_id = f"sample-bad-end-{idx}"
            sample_copy.source_seed = f"SE-{idx}"
            sample_copy.state_hash = f"hash-end-{idx}"
            on_policy_pool.append(sample_copy)

        final_samples = _assemble_dataset(
            on_policy_samples=on_policy_pool,
            historical_samples=[],
            motif_samples=[bash_sample, lethal_sample, turn_lethal_sample, bad_end_sample],
            target_samples=32,
            rng_seed=0,
        )

        assert sum(1 for sample in final_samples if "bash_before_strike" in sample.motif_labels) >= 2
        assert sum(1 for sample in final_samples if "direct_lethal_first_action" in sample.motif_labels) >= 2
        assert sum(1 for sample in final_samples if "turn_lethal_no_end_turn" in sample.motif_labels) >= 2
        assert sum(1 for sample in final_samples if sample.source_bucket == "on_policy") >= 4
        assert len(final_samples) == 32

    def test_dataset_assembly_reserves_holdout_anchor_coverage(self):
        from build_combat_teacher_dataset import _assemble_dataset
        from combat_teacher_dataset import CombatTeacherSample, sample_metric_applicable

        bash_state = _make_teacher_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=20,
        )
        bash_actions = [
            {"action": "play_card", "card_index": 0, "label": "Bash", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]
        train_bash = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="train-bash",
            split="train",
            source_bucket="historical",
            source_seed="TB",
            source_checkpoint="combat.pt",
            state_hash="hash-train-bash",
            motif_labels=["bash_before_strike", "bad_end_turn"],
            state=bash_state,
            legal_actions=bash_actions,
            baseline_logits=[0.1, 0.2, 0.0],
            baseline_probs=[0.34, 0.38, 0.28],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "BASH"}],
            per_action_score=[1.0, 0.5, -0.2],
            per_action_regret=[0.0, 0.5, 1.2],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 0.7, "expected_hp_loss": 4.0, "expected_potion_cost": 0.0},
        )
        holdout_bash = copy.deepcopy(train_bash)
        holdout_bash.sample_id = "holdout-bash"
        holdout_bash.split = "holdout"
        holdout_bash.source_seed = "HB"
        holdout_bash.state_hash = "hash-holdout-bash"

        on_policy_pool = []
        for idx in range(10):
            sample_copy = copy.deepcopy(train_bash)
            sample_copy.sample_id = f"train-bash-{idx}"
            sample_copy.source_seed = f"TB-{idx}"
            sample_copy.state_hash = f"hash-train-bash-{idx}"
            sample_copy.motif_labels = ["bad_end_turn"]
            sample_copy.best_action_index = 1
            sample_copy.per_action_score = [0.1, 0.5, -0.2]
            sample_copy.per_action_regret = [0.4, 0.0, 0.7]
            on_policy_pool.append(sample_copy)

        final_samples = _assemble_dataset(
            on_policy_samples=on_policy_pool,
            historical_samples=[train_bash, holdout_bash],
            motif_samples=[train_bash, holdout_bash],
            target_samples=16,
            rng_seed=0,
        )

        holdout_bash_count = sum(
            1 for sample in final_samples
            if sample.split == "holdout" and sample_metric_applicable(sample, "bash_before_strike")
        )
        assert holdout_bash_count >= 1

    def test_dataset_assembly_can_zero_historical_bulk_fraction(self):
        from build_combat_teacher_dataset import _assemble_dataset
        from combat_teacher_dataset import CombatTeacherSample

        base_state = _make_teacher_combat_state(
            [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
            enemy_hp=30,
        )
        base_actions = [
            {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        def _sample(sample_id: str, bucket: str) -> CombatTeacherSample:
            return CombatTeacherSample(
                schema_version="combat_teacher_dataset.v1",
                sample_id=sample_id,
                split="train",
                source_bucket=bucket,
                source_seed=sample_id,
                source_checkpoint="combat.pt",
                state_hash=f"hash-{sample_id}",
                motif_labels=["bad_end_turn"],
                state=base_state,
                legal_actions=base_actions,
                baseline_logits=[0.1, 0.2],
                baseline_probs=[0.45, 0.55],
                baseline_best_action_index=1,
                best_action_index=0,
                best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
                per_action_score=[0.5, -0.2],
                per_action_regret=[0.0, 0.7],
                root_value=0.5,
                leaf_breakdown={"total": 0.5},
                continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
            )

        on_policy_samples = [_sample(f"on-{idx}", "on_policy") for idx in range(12)]
        historical_samples = [_sample(f"hist-{idx}", "historical") for idx in range(12)]

        final_samples = _assemble_dataset(
            on_policy_samples=on_policy_samples,
            historical_samples=historical_samples,
            motif_samples=[],
            target_samples=10,
            rng_seed=0,
            historical_final_fraction=0.0,
        )

        non_anchor_historical = sum(1 for sample in final_samples if sample.source_bucket == "historical")
        assert non_anchor_historical == 0

    def test_targeted_hard_motif_selection_prefers_high_regret_and_diverse_hashes(self):
        from build_combat_teacher_dataset import _take_prioritized_samples
        from combat_teacher_dataset import CombatTeacherSample

        base_state = _make_teacher_combat_state(
            [
                {"id": "BASH", "name": "Bash", "cost": 2, "is_upgraded": False},
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=18,
        )
        base_actions = [
            {"action": "play_card", "card_index": 0, "label": "BASH", "card_id": "BASH", "target_id": "cultist-0", "is_enabled": True},
            {"action": "play_card", "card_index": 1, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        def _sample(sample_id: str, *, bucket: str, regret: float, state_hash: str) -> CombatTeacherSample:
            return CombatTeacherSample(
                schema_version="combat_teacher_dataset.v1",
                sample_id=sample_id,
                split="train",
                source_bucket=bucket,
                source_seed=sample_id,
                source_checkpoint="combat.pt",
                state_hash=state_hash,
                motif_labels=["bash_before_strike", "bad_end_turn"],
                state=base_state,
                legal_actions=base_actions,
                baseline_logits=[0.1, 0.2, 0.0],
                baseline_probs=[0.34, 0.38, 0.28],
                baseline_best_action_index=1,
                best_action_index=0,
                best_full_turn_line=[{"action": "play_card", "card_id": "BASH"}],
                per_action_score=[1.0, 1.0 - regret, -0.2],
                per_action_regret=[0.0, regret, 1.2],
                root_value=1.0,
                leaf_breakdown={"total": 1.0},
                continuation_targets={"win_prob": 0.7, "expected_hp_loss": 4.0, "expected_potion_cost": 0.0},
            )

        pool = [
            _sample("on-low", bucket="on_policy", regret=0.2, state_hash="hash-on"),
            _sample("hist-hi-a", bucket="historical", regret=0.9, state_hash="hash-h1"),
            _sample("hist-hi-b-samehash", bucket="historical", regret=0.85, state_hash="hash-h1"),
            _sample("hist-hi-c", bucket="historical", regret=0.8, state_hash="hash-h2"),
        ]

        selected = _take_prioritized_samples(
            pool,
            2,
            selected_ids=set(),
            motif_name="bash_before_strike",
            prefer_diverse_state_hash=True,
        )

        assert [sample.sample_id for sample in selected] == ["hist-hi-a", "hist-hi-c"]

    def test_dataset_assembly_prioritizes_on_policy_holdout_for_direct_lethal(self):
        from build_combat_teacher_dataset import _assemble_dataset
        from combat_teacher_dataset import CombatTeacherSample, sample_metric_applicable

        lethal_state = _make_teacher_combat_state(
            [
                {"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False},
            ],
            enemy_hp=6,
        )
        lethal_actions = [
            {"action": "play_card", "card_index": 0, "label": "Strike", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
            {"action": "end_turn", "is_enabled": True},
        ]

        def _sample(sample_id: str, *, bucket: str, split: str, baseline_regret: float) -> CombatTeacherSample:
            return CombatTeacherSample(
                schema_version="combat_teacher_dataset.v1",
                sample_id=sample_id,
                split=split,
                source_bucket=bucket,
                source_seed=sample_id,
                source_checkpoint="combat.pt",
                state_hash=f"hash-{sample_id}",
                motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
                state=lethal_state,
                legal_actions=lethal_actions,
                baseline_logits=[0.1, 0.3],
                baseline_probs=[0.45, 0.55],
                baseline_best_action_index=1,
                best_action_index=0,
                best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
                per_action_score=[1.0, -0.5],
                per_action_regret=[0.0, baseline_regret],
                root_value=1.0,
                leaf_breakdown={"total": 1.0},
                continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
            )

        on_policy_holdout = _sample("holdout-on-policy", bucket="on_policy", split="holdout", baseline_regret=0.2)
        historical_holdout_a = _sample("holdout-historical-a", bucket="historical", split="holdout", baseline_regret=2.0)
        historical_holdout_b = _sample("holdout-historical-b", bucket="historical", split="holdout", baseline_regret=1.9)
        train_fill = [
            _sample(f"train-on-{idx}", bucket="on_policy", split="train", baseline_regret=0.5)
            for idx in range(8)
        ]

        final_samples = _assemble_dataset(
            on_policy_samples=[on_policy_holdout, *train_fill],
            historical_samples=[historical_holdout_a, historical_holdout_b],
            motif_samples=[],
            target_samples=16,
            rng_seed=0,
            historical_final_fraction=0.0,
        )

        holdout_direct_lethal = [
            sample
            for sample in final_samples
            if sample.split == "holdout" and sample_metric_applicable(sample, "direct_lethal_first_action")
        ]
        assert any(sample.source_bucket == "on_policy" for sample in holdout_direct_lethal)

    def test_regression_motif_samples_cover_sparse_patterns(self):
        from combat_teacher_regression_samples import build_regression_motif_samples

        samples = build_regression_motif_samples()
        labels = {label for sample in samples for label in sample.motif_labels}

        assert "bash_before_strike" in labels
        assert "missed_lethal" in labels
        assert "direct_lethal_first_action" in labels
        assert "turn_lethal_no_end_turn" in labels
        assert "bodyslam_before_block" in labels
        assert "potion_misuse" in labels

    def test_regression_motif_samples_have_train_and_holdout_coverage(self):
        from combat_teacher_dataset import sample_metric_applicable
        from combat_teacher_regression_samples import build_regression_motif_samples

        samples = build_regression_motif_samples()
        motifs = (
            "bash_before_strike",
            "direct_lethal_first_action",
            "turn_lethal_no_end_turn",
            "bodyslam_before_block",
            "potion_misuse",
        )
        for motif in motifs:
            train_count = sum(
                1
                for sample in samples
                if sample.split == "train" and sample_metric_applicable(sample, motif)
            )
            holdout_count = sum(
                1
                for sample in samples
                if sample.split == "holdout" and sample_metric_applicable(sample, motif)
            )
            assert train_count >= 2, f"{motif} train coverage too small: {train_count}"
            assert holdout_count >= 2, f"{motif} holdout coverage too small: {holdout_count}"

    def test_teacher_train_weights_simple_missed_lethal_above_regular_states(self):
        from combat_teacher_dataset import CombatTeacherSample
        from train_combat_teacher import _train_sample_weight

        regular = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="regular",
            split="train",
            source_bucket="on_policy",
            source_seed="S0",
            source_checkpoint="combat.pt",
            state_hash="hash-regular",
            motif_labels=["bad_end_turn"],
            state=_make_teacher_combat_state(
                [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
                enemy_hp=30,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[0.5, -0.2],
            per_action_regret=[0.0, 0.7],
            root_value=0.5,
            leaf_breakdown={"total": 0.5},
            continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
        )
        simple_lethal = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="simple-lethal",
            split="train",
            source_bucket="motif_regression",
            source_seed="S1",
            source_checkpoint="regression",
            state_hash="hash-simple-lethal",
            motif_labels=["missed_lethal", "direct_lethal_first_action", "bad_end_turn"],
            state=_make_teacher_combat_state(
                [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.3],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, -0.5],
            per_action_regret=[0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )

        regular_weight = _train_sample_weight(
            regular,
            missed_lethal_weight=2.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=2.0,
            regression_weight=1.5,
            baseline_regret_weight_scale=0.0,
        )
        lethal_weight = _train_sample_weight(
            simple_lethal,
            missed_lethal_weight=2.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=2.0,
            regression_weight=1.5,
            baseline_regret_weight_scale=0.0,
        )

        assert lethal_weight > regular_weight

    def test_teacher_train_weights_direct_lethal_above_regular_states(self):
        from combat_teacher_dataset import CombatTeacherSample
        from train_combat_teacher import _train_sample_weight

        regular = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="regular-direct",
            split="train",
            source_bucket="on_policy",
            source_seed="SD0",
            source_checkpoint="combat.pt",
            state_hash="hash-regular-direct",
            motif_labels=["bad_end_turn"],
            state=_make_teacher_combat_state(
                [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
                enemy_hp=30,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[0.5, -0.2],
            per_action_regret=[0.0, 0.7],
            root_value=0.5,
            leaf_breakdown={"total": 0.5},
            continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
        )
        direct_lethal = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="direct-lethal",
            split="train",
            source_bucket="on_policy",
            source_seed="SD1",
            source_checkpoint="combat.pt",
            state_hash="hash-direct-lethal",
            motif_labels=["direct_lethal_first_action", "missed_lethal"],
            state=_make_teacher_combat_state(
                [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.3],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, -0.5],
            per_action_regret=[0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )

        regular_weight = _train_sample_weight(
            regular,
            missed_lethal_weight=1.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=1.0,
            regression_weight=1.0,
            baseline_regret_weight_scale=0.0,
        )
        direct_lethal_weighted = _train_sample_weight(
            direct_lethal,
            missed_lethal_weight=1.0,
            direct_lethal_weight=2.0,
            simple_missed_lethal_extra_weight=1.0,
            regression_weight=1.0,
            baseline_regret_weight_scale=0.0,
        )

        assert direct_lethal_weighted > regular_weight

    def test_teacher_baseline_anchor_weight_only_boosts_direct_lethal_when_baseline_is_correct(self):
        from combat_teacher_dataset import CombatTeacherSample
        from train_combat_teacher import _baseline_anchor_weight

        baseline_correct = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="baseline-correct-direct-lethal",
            split="train",
            source_bucket="on_policy",
            source_seed="BA0",
            source_checkpoint="combat.pt",
            state_hash="hash-baseline-correct-direct-lethal",
            motif_labels=["direct_lethal_first_action", "missed_lethal"],
            state=_make_teacher_combat_state(
                [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.2, 0.1],
            baseline_probs=[0.6, 0.4],
            baseline_best_action_index=0,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, -0.5],
            per_action_regret=[0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )
        baseline_wrong = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="baseline-wrong-direct-lethal",
            split="train",
            source_bucket="historical",
            source_seed="BA1",
            source_checkpoint="combat.pt",
            state_hash="hash-baseline-wrong-direct-lethal",
            motif_labels=["direct_lethal_first_action", "missed_lethal"],
            state=_make_teacher_combat_state(
                [{"id": "STRIKE_IRONCLAD", "name": "Strike", "cost": 1, "is_upgraded": False}],
                enemy_hp=6,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "STRIKE_IRONCLAD", "card_id": "STRIKE_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.3],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "STRIKE_IRONCLAD"}],
            per_action_score=[1.0, -0.5],
            per_action_regret=[0.0, 1.5],
            root_value=1.0,
            leaf_breakdown={"total": 1.0},
            continuation_targets={"win_prob": 1.0, "expected_hp_loss": 0.0, "expected_potion_cost": 0.0},
        )

        correct_weight = _baseline_anchor_weight(
            baseline_correct,
            direct_lethal_baseline_anchor_weight=2.0,
        )
        wrong_weight = _baseline_anchor_weight(
            baseline_wrong,
            direct_lethal_baseline_anchor_weight=2.0,
        )

        assert correct_weight == 2.0
        assert wrong_weight == 1.0

    def test_teacher_score_kl_ignores_masked_actions(self):
        from train_combat_teacher import _masked_policy_kl_to_reference

        masked_scores = torch.tensor([[np.log(0.7), np.log(0.3), -1e9]], dtype=torch.float32)
        baseline_probs = torch.tensor([[0.7, 0.3, 0.9]], dtype=torch.float32)
        action_mask = torch.tensor([[True, True, False]], dtype=torch.bool)

        kl = _masked_policy_kl_to_reference(masked_scores, baseline_probs, action_mask)

        assert kl.shape == (1,)
        assert float(kl.item()) == pytest.approx(0.0, abs=1e-6)

    def test_teacher_train_weights_can_scale_with_baseline_regret(self):
        from combat_teacher_dataset import CombatTeacherSample
        from train_combat_teacher import _train_sample_weight

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="regret-weighted",
            split="train",
            source_bucket="on_policy",
            source_seed="S2",
            source_checkpoint="combat.pt",
            state_hash="hash-regret-weighted",
            motif_labels=["bad_end_turn"],
            state=_make_teacher_combat_state(
                [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
                enemy_hp=30,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[0.5, -0.2],
            per_action_regret=[0.0, 0.7],
            root_value=0.5,
            leaf_breakdown={"total": 0.5},
            continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
        )

        base_weight = _train_sample_weight(
            sample,
            missed_lethal_weight=1.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=1.0,
            regression_weight=1.0,
            baseline_regret_weight_scale=0.0,
        )
        regret_weighted = _train_sample_weight(
            sample,
            missed_lethal_weight=1.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=1.0,
            regression_weight=1.0,
            baseline_regret_weight_scale=2.0,
        )

        assert regret_weighted > base_weight

    def test_teacher_train_ignores_sentinel_baseline_regret_for_weighting(self):
        from combat_teacher_dataset import CombatTeacherSample
        from train_combat_teacher import _train_sample_weight

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sentinel-regret",
            split="train",
            source_bucket="on_policy",
            source_seed="S3",
            source_checkpoint="combat.pt",
            state_hash="hash-sentinel-regret",
            motif_labels=["bad_end_turn"],
            state=_make_teacher_combat_state(
                [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
                enemy_hp=30,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[0.5, -0.2],
            per_action_regret=[0.0, 1e9],
            root_value=0.5,
            leaf_breakdown={"total": 0.5},
            continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
        )

        base_weight = _train_sample_weight(
            sample,
            missed_lethal_weight=1.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=1.0,
            regression_weight=1.0,
            baseline_regret_weight_scale=0.0,
        )
        regret_weighted = _train_sample_weight(
            sample,
            missed_lethal_weight=1.0,
            direct_lethal_weight=1.0,
            simple_missed_lethal_extra_weight=1.0,
            regression_weight=1.0,
            baseline_regret_weight_scale=2.0,
        )

        assert regret_weighted == base_weight

    def test_microbench_caps_sentinel_regret_values(self):
        from combat_microbench import _sample_regret
        from combat_teacher_dataset import CombatTeacherSample

        sample = CombatTeacherSample(
            schema_version="combat_teacher_dataset.v1",
            sample_id="sentinel-microbench",
            split="holdout",
            source_bucket="on_policy",
            source_seed="SM",
            source_checkpoint="combat.pt",
            state_hash="hash-sentinel-microbench",
            motif_labels=["bad_end_turn"],
            state=_make_teacher_combat_state(
                [{"id": "DEFEND_IRONCLAD", "name": "Defend", "cost": 1, "is_upgraded": False}],
                enemy_hp=30,
            ),
            legal_actions=[
                {"action": "play_card", "card_index": 0, "label": "DEFEND_IRONCLAD", "card_id": "DEFEND_IRONCLAD", "target_id": "cultist-0", "is_enabled": True},
                {"action": "end_turn", "is_enabled": True},
            ],
            baseline_logits=[0.1, 0.2],
            baseline_probs=[0.45, 0.55],
            baseline_best_action_index=1,
            best_action_index=0,
            best_full_turn_line=[{"action": "play_card", "card_id": "DEFEND_IRONCLAD"}],
            per_action_score=[0.5, -0.2],
            per_action_regret=[0.0, 1e9],
            root_value=0.5,
            leaf_breakdown={"total": 0.5},
            continuation_targets={"win_prob": 0.5, "expected_hp_loss": 6.0, "expected_potion_cost": 0.0},
        )

        assert _sample_regret(sample, 1) == 1.0


# ---------------------------------------------------------------------------
# Task-0 parity harness tests
# ---------------------------------------------------------------------------

class TestCombatTurnTrace:
    """Smoke tests for combat_turn_trace.py — trace recording and comparison."""

    def _make_combat_state(
        self,
        *,
        hp: int = 70,
        block: int = 0,
        energy: int = 3,
        hand: list[str] | None = None,
        enemy_hp: int = 30,
        round_number: int = 1,
        state_type: str = "monster",
    ) -> dict:
        hand = hand or ["Strike", "Strike", "Defend", "Defend", "Bash"]
        hand_cards = [{"id": c, "label": c, "cost": 1, "type": "Attack", "upgrades": 0} for c in hand]
        return {
            "state_type": state_type,
            "terminal": False,
            "run_outcome": None,
            "run": {"floor": 3, "act": 1},
            "battle": {
                "round_number": round_number,
                "draw_pile_count": 5,
                "discard_pile_count": 0,
                "exhaust_pile_count": 0,
                "player": {
                    "hp": hp,
                    "max_hp": 80,
                    "block": block,
                    "energy": energy,
                    "hand": hand_cards,
                    "deck": [],
                    "relics": [],
                    "potions": [],
                    "status": [],
                    "draw_pile_count": 5,
                    "discard_pile_count": 0,
                    "exhaust_pile_count": 0,
                    "gold": 99,
                    "open_potion_slots": 2,
                },
                "enemies": [
                    {
                        "id": "Cultist",
                        "name": "Cultist",
                        "hp": enemy_hp,
                        "max_hp": 50,
                        "block": 0,
                        "is_alive": True,
                        "intents": [{"type": "Attack"}],
                        "status": [],
                    }
                ],
            },
            "player": {
                "hp": hp,
                "max_hp": 80,
                "block": block,
                "gold": 99,
                "open_potion_slots": 2,
                "deck": [],
                "relics": [],
                "potions": [],
                "status": [],
                "hand": hand_cards,
            },
            "legal_actions": [
                {"action": "play_card", "index": 0, "card_index": 0, "label": "Strike", "is_enabled": True},
                {"action": "play_card", "index": 1, "card_index": 1, "label": "Strike", "is_enabled": True},
                {"action": "end_turn", "index": 5, "is_enabled": True},
            ],
        }

    def test_trace_entry_records_correct_fields(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        tracer = CombatTurnTracer()
        state = self._make_combat_state()
        action = {"action": "play_card", "label": "Strike", "card_index": 0}

        entry = tracer.record(state, action)

        assert entry.step == 0
        assert entry.turn == 1
        assert entry.player_hp == 70
        assert entry.energy == 3
        assert entry.draw_count == 5
        assert entry.legal_action_count == 3
        assert len(entry.legal_mask) == 3
        assert entry.state_type == "monster"
        assert entry.floor == 3
        assert entry.act == 1
        assert "Strike" in entry.hand_labels
        assert len(entry.enemies) == 1
        assert entry.enemies[0]["name"] == "Cultist"
        assert entry.public_state_hash  # non-empty

    def test_public_state_trace_builds_hash_and_mask(self):
        build_trace_entry = _import_or_skip("public_state_trace").build_trace_entry

        state = self._make_combat_state()
        entry = build_trace_entry(state, step=0, action={"action": "end_turn", "index": 5})

        assert entry.public_state_hash
        assert entry.legal_mask[0][0] == "play_card"
        assert entry.legal_mask[-1][0] == "end_turn"
        assert entry.action["action"] == "end_turn"

    def test_tracer_increments_step(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        tracer = CombatTurnTracer()
        state = self._make_combat_state()
        action = {"action": "play_card", "label": "Strike", "card_index": 0}

        e0 = tracer.record(state, action)
        e1 = tracer.record(state, action)
        assert e0.step == 0
        assert e1.step == 1
        assert len(tracer.entries) == 2

    def test_tracer_reset_clears(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        tracer = CombatTurnTracer()
        tracer.record(self._make_combat_state(), {"action": "end_turn"})
        assert len(tracer.entries) == 1
        tracer.reset()
        assert len(tracer.entries) == 0

    def test_compare_traces_identical(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        tracer = CombatTurnTracer()
        state = self._make_combat_state()
        tracer.record(state, {"action": "play_card", "label": "Strike"})
        tracer.record(state, {"action": "end_turn"})
        trace = tracer.entries

        mismatches = CombatTurnTracer.compare_traces(trace, trace)
        assert len(mismatches) == 0

    def test_compare_traces_detects_hp_diff(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        t_a = CombatTurnTracer()
        t_b = CombatTurnTracer()

        state_a = self._make_combat_state(hp=70)
        state_b = self._make_combat_state(hp=65)

        t_a.record(state_a, {"action": "end_turn"})
        t_b.record(state_b, {"action": "end_turn"})

        mismatches = CombatTurnTracer.compare_traces(t_a.entries, t_b.entries)
        field_names = [m.field for m in mismatches]
        assert "player_hp" in field_names

    def test_compare_traces_detects_legal_mask_diff(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        t_a = CombatTurnTracer()
        t_b = CombatTurnTracer()

        state_a = self._make_combat_state()
        state_b = self._make_combat_state()
        state_b["legal_actions"] = [
            {"action": "play_card", "index": 0, "card_index": 0, "label": "Strike", "is_enabled": True},
            {"action": "end_turn", "index": 5, "is_enabled": True},
        ]

        t_a.record(state_a, {"action": "end_turn"})
        t_b.record(state_b, {"action": "end_turn"})

        mismatches = CombatTurnTracer.compare_traces(t_a.entries, t_b.entries)
        field_names = [m.field for m in mismatches]
        assert "legal_mask" in field_names

    def test_compare_traces_detects_length_mismatch(self):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        t_a = CombatTurnTracer()
        t_b = CombatTurnTracer()

        state = self._make_combat_state()
        t_a.record(state, {"action": "end_turn"})
        t_a.record(state, {"action": "end_turn"})
        t_b.record(state, {"action": "end_turn"})

        mismatches = CombatTurnTracer.compare_traces(t_a.entries, t_b.entries)
        assert any(m.field == "trace_length" for m in mismatches)

    def test_flush_and_load_roundtrip(self, tmp_path):
        CombatTurnTracer = _import_or_skip("combat_turn_trace").CombatTurnTracer

        tracer = CombatTurnTracer()
        state = self._make_combat_state()
        tracer.record(state, {"action": "play_card", "label": "Strike"})
        tracer.record(state, {"action": "end_turn"})

        path = tmp_path / "trace.jsonl"
        tracer.flush(path)

        loaded = CombatTurnTracer.load_trace(path)
        assert len(loaded) == 2
        assert loaded[0].step == 0
        assert loaded[1].step == 1

        # Roundtrip comparison should show no mismatches
        mismatches = CombatTurnTracer.compare_traces(tracer.entries, loaded)
        assert len(mismatches) == 0


class TestSaveLoadParityReport:
    """Smoke tests for saveload_combat_parity.py — report generation on mock data."""

    def test_parity_result_to_dict(self):
        ParityResult = _import_or_skip("saveload_combat_parity").ParityResult

        r = ParityResult(
            test_case="immediate_roundtrip",
            seed="TEST_001",
            floor=3,
            turn=1,
            verdict="exact",
            matching_fields=["state_type", "player", "enemies"],
            mismatched_fields=[],
            hand_order_preserved=True,
            hand_labels_preserved=True,
            legal_actions_preserved=True,
            details={},
        )
        d = r.to_dict()
        assert d["test_case"] == "immediate_roundtrip"
        assert d["verdict"] == "exact"
        assert d["hand_order_preserved"] is True

    def test_build_report_summary(self):
        saveload_combat_parity = _import_or_skip("saveload_combat_parity")
        ParityResult = saveload_combat_parity.ParityResult
        build_report = saveload_combat_parity.build_report

        results = [
            ParityResult("immediate_roundtrip", "S1", 3, 1, "exact", ["a"], [], True, True, True, {}),
            ParityResult("mid_turn", "S1", 3, 1, "resumable", ["a"], ["hand"], False, True, True, {}),
            ParityResult("rollback", "S1", 3, 1, "exact", ["a"], [], True, True, True, {}),
            ParityResult("idempotency", "S1", 3, 1, "exact", ["a"], [], True, True, True, {}),
        ]
        report = build_report(results)

        s = report["summary"]
        assert s["total_tests"] == 4
        assert s["exact"] == 3
        assert s["resumable"] == 1
        assert s["hand_labels_preserved_rate"] == 1.0
        assert s["legal_actions_preserved_rate"] == 1.0

    def test_build_report_mcts_feasibility_needs_work(self):
        saveload_combat_parity = _import_or_skip("saveload_combat_parity")
        ParityResult = saveload_combat_parity.ParityResult
        build_report = saveload_combat_parity.build_report

        results = [
            ParityResult("immediate_roundtrip", "S1", 3, 1, "diverged", [], ["all"], False, False, False, {}),
        ]
        report = build_report(results)
        assert report["summary"]["mcts_feasibility"] == "NEEDS_WORK"

    def test_extract_hand_helpers(self):
        saveload_combat_parity = _import_or_skip("saveload_combat_parity")
        _extract_hand_ordered = saveload_combat_parity._extract_hand_ordered
        _extract_hand_labels_sorted = saveload_combat_parity._extract_hand_labels_sorted

        state = {
            "battle": {
                "player": {
                    "hand": [
                        {"id": "Defend", "label": "Defend", "cost": 1},
                        {"id": "Strike", "label": "Strike", "cost": 1},
                        {"id": "Bash", "label": "Bash", "cost": 2},
                    ]
                }
            }
        }

        ordered = _extract_hand_ordered(state)
        assert ordered == [("Defend", 1), ("Strike", 1), ("Bash", 2)]

        labels = _extract_hand_labels_sorted(state)
        assert labels == ["Bash", "Defend", "Strike"]


# ---------------------------------------------------------------------------
# Ranking loss + matchup infrastructure tests
# ---------------------------------------------------------------------------

class TestRankingLoss:
    """Smoke tests for ranking_loss.py — listwise and pairwise losses."""

    def test_listwise_identical_scores_low_loss(self):
        from ranking_loss import listwise_ranking_loss

        scores = torch.tensor([[1.0, 0.5, 0.2, -0.1]])
        mask = torch.tensor([[True, True, True, True]])

        loss = listwise_ranking_loss(scores, scores, mask)
        assert loss.item() < 0.01  # nearly zero for identical inputs

    def test_listwise_opposite_scores_high_loss(self):
        from ranking_loss import listwise_ranking_loss

        predicted = torch.tensor([[1.0, 0.5, 0.2, -0.1]])
        target = torch.tensor([[-0.1, 0.2, 0.5, 1.0]])  # reversed
        mask = torch.tensor([[True, True, True, True]])

        loss = listwise_ranking_loss(predicted, target, mask)
        assert loss.item() > 0.1  # significant loss for opposite rankings

    def test_listwise_respects_mask(self):
        from ranking_loss import listwise_ranking_loss

        predicted = torch.tensor([[1.0, 0.5, 999.0, -999.0]])
        target = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False, False]])

        loss = listwise_ranking_loss(predicted, target, mask)
        assert loss.isfinite()
        assert loss.item() < 0.1  # masked entries don't affect loss

    def test_pairwise_correct_order_low_loss(self):
        from ranking_loss import pairwise_ranking_loss

        # Predicted matches target ordering
        predicted = torch.tensor([[3.0, 2.0, 1.0]])
        target = torch.tensor([[1.0, 0.5, 0.0]])
        mask = torch.tensor([[True, True, True]])

        loss = pairwise_ranking_loss(predicted, target, mask)
        assert loss.item() < 0.5  # correct ordering has low loss

    def test_pairwise_wrong_order_high_loss(self):
        from ranking_loss import pairwise_ranking_loss

        # Predicted reverses target ordering
        predicted = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 0.5, 0.0]])
        mask = torch.tensor([[True, True, True]])

        loss = pairwise_ranking_loss(predicted, target, mask)
        assert loss.item() > 0.5  # wrong ordering has high loss

    def test_pairwise_empty_returns_zero(self):
        from ranking_loss import pairwise_ranking_loss

        predicted = torch.tensor([[1.0]])
        target = torch.tensor([[0.5]])
        mask = torch.tensor([[True]])

        loss = pairwise_ranking_loss(predicted, target, mask)
        assert loss.item() == 0.0  # can't form pairs with 1 element

    def test_loss_gradient_flows(self):
        from ranking_loss import listwise_ranking_loss

        predicted = torch.tensor([[1.0, 0.5, 0.2]], requires_grad=True)
        target = torch.tensor([[-0.1, 0.2, 1.0]])
        mask = torch.tensor([[True, True, True]])

        loss = listwise_ranking_loss(predicted, target, mask)
        loss.backward()
        assert predicted.grad is not None
        assert predicted.grad.abs().sum() > 0


class TestMatchupScoreHead:
    """Smoke tests for matchup_score_head in FullRunPolicyNetworkV2."""

    def test_matchup_head_exists_and_zero_init(self):
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from vocab import load_vocab

        vocab = load_vocab()
        net = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=32)

        # Check head exists
        assert hasattr(net, "matchup_score_head")
        assert hasattr(net, "compute_matchup_scores")

        # Final layer should be zero-initialized
        final_layer = net.matchup_score_head[-1]
        assert final_layer.weight.abs().max().item() == 0.0
        assert final_layer.bias.abs().max().item() == 0.0

    def test_matchup_head_param_count_reasonable(self):
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from vocab import load_vocab

        vocab = load_vocab()
        net = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=32)

        matchup_params = sum(p.numel() for p in net.matchup_score_head.parameters())
        total_params = net.param_count()

        # Matchup head should be small relative to total network
        assert matchup_params < total_params * 0.05  # less than 5%
        assert matchup_params > 0

    def test_old_checkpoint_loads_with_new_head(self):
        """Verify that loading old checkpoint (without matchup_score_head) works via strict=False."""
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from vocab import load_vocab

        vocab = load_vocab()
        net_old = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=32)
        # Simulate old checkpoint by removing matchup head keys
        sd = net_old.state_dict()
        old_sd = {k: v for k, v in sd.items() if "matchup_score" not in k}

        net_new = FullRunPolicyNetworkV2(vocab=vocab, embed_dim=32)
        # Should not raise with strict=False
        missing, unexpected = net_new.load_state_dict(old_sd, strict=False)
        assert any("matchup_score" in k for k in missing)
        assert len(unexpected) == 0


class TestMatchupDataset:
    """Smoke tests for matchup_dataset.py — dataset loading."""

    def test_empty_dataset(self):
        from matchup_dataset import MatchupRankingDataset
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = MatchupRankingDataset(tmpdir)
            assert len(ds) == 0
            assert ds.sample_batch(8) is None

    def test_load_jsonl(self, tmp_path):
        from matchup_dataset import MatchupRankingDataset

        # Write mock data
        data = [
            {"scores": [0.8, 0.5, 0.3, 0.2], "best_idx": 0, "options": [{}, {}, {}, {}]},
            {"scores": [0.4, 0.9, 0.6], "best_idx": 1, "options": [{}, {}, {}]},
        ]
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        ds = MatchupRankingDataset(str(tmp_path))
        assert len(ds) == 2

    def test_sample_batch_shapes(self, tmp_path):
        from matchup_dataset import MatchupRankingDataset, MAX_OPTIONS

        data = [
            {"scores": [0.8, 0.5, 0.3, 0.2], "best_idx": 0, "options": [{}, {}, {}, {}]},
            {"scores": [0.4, 0.9, 0.6], "best_idx": 1, "options": [{}, {}, {}]},
            {"scores": [0.1, 0.2], "best_idx": 1, "options": [{}, {}]},
        ]
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        ds = MatchupRankingDataset(str(tmp_path))
        batch = ds.sample_batch(4)
        assert batch is not None
        assert batch["target_scores"].shape == (4, MAX_OPTIONS)
        assert batch["option_mask"].shape == (4, MAX_OPTIONS)
        assert batch["option_mask"].dtype == torch.bool

    def test_dataset_stats(self, tmp_path):
        from matchup_dataset import MatchupRankingDataset

        data = [
            {"scores": [0.8, 0.5, 0.3, 0.2], "best_idx": 0},
            {"scores": [0.4, 0.9, 0.6, 0.1], "best_idx": 1},
        ]
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        ds = MatchupRankingDataset(str(tmp_path))
        stats = ds.get_stats()
        assert stats["num_samples"] == 2
        assert stats["avg_options"] == 4.0


class TestGenerateCardRanking:
    """Smoke tests for generate_card_ranking_data.py — scoring logic."""

    def test_compute_option_scores(self):
        from generate_card_ranking_data import CombatOutcome, compute_option_scores

        outcomes = {
            0: CombatOutcome(won=True, hp_after=70, hp_lost=10, turns=3, terminal_state_type="combat_rewards"),
            1: CombatOutcome(won=True, hp_after=50, hp_lost=30, turns=5, terminal_state_type="combat_rewards"),
            2: CombatOutcome(won=False, hp_after=0, hp_lost=80, turns=8, terminal_state_type="game_over"),
        }
        scores = compute_option_scores(outcomes, max_hp=80)

        assert len(scores) == 3
        assert scores[0] > scores[1]  # less HP lost
        assert scores[1] > scores[2]  # won vs lost
        assert scores[2] < 0.5  # lost fight

    def test_compute_option_scores_all_wins(self):
        from generate_card_ranking_data import CombatOutcome, compute_option_scores

        outcomes = {
            0: CombatOutcome(won=True, hp_after=75, hp_lost=5, turns=2, terminal_state_type="cr"),
            1: CombatOutcome(won=True, hp_after=60, hp_lost=20, turns=4, terminal_state_type="cr"),
        }
        scores = compute_option_scores(outcomes, max_hp=80)
        assert scores[0] > scores[1]  # better HP + fewer turns

    def test_extract_card_reward_options_format(self):
        from generate_card_ranking_data import _extract_card_reward_options

        state = {
            "card_reward": {
                "cards": [
                    {"id": "pommel_strike", "name": "Pommel Strike", "index": 0},
                    {"id": "shrug_it_off", "name": "Shrug It Off", "index": 1},
                    {"id": "carnage", "name": "Carnage", "index": 2},
                ]
            },
            "legal_actions": [
                {"action": "select_card_reward", "index": 0},
                {"action": "select_card_reward", "index": 1},
                {"action": "select_card_reward", "index": 2},
                {"action": "skip_card_reward", "index": 3},
            ],
        }
        options = _extract_card_reward_options(state)
        assert len(options) == 4
        assert options[0]["card_id"] == "pommel_strike"
        assert options[3]["type"] == "skip"

    def test_safe_load_state_dict_accepts_current_and_legacy_keys(self):
        import torch.nn as nn
        from generate_card_ranking_data import _safe_load_state_dict

        model = nn.Linear(4, 2)
        original = model.weight.detach().clone()
        replacement = {
            "weight": original + 1.0,
            "bias": model.bias.detach().clone() + 2.0,
            "unused": model.bias.detach().clone(),
        }
        _safe_load_state_dict(model, replacement)
        assert torch.allclose(model.weight, original + 1.0)
        assert torch.allclose(model.bias, replacement["bias"])


class TestIntegrationPanelRunner:
    def test_compact_metrics_keeps_expected_fields(self):
        _compact_metrics = _import_or_skip("run_integration_panel")._compact_metrics

        payload = {
            "summaries": {
                "nn": {
                    "avg_floor": 9.1,
                    "boss_reach_rate": 0.1,
                    "act1_clear_rate": 0.0,
                    "avg_boss_hp_fraction_dealt": 0.37,
                    "error_count": 0,
                    "timeout_count": 1,
                    "action_source_counts": {"nn": 100, "auto_progress": 12},
                    "card_reward_skip_rate_by_boss": {"slime": 0.25},
                    "combat_teacher_override_counts": {"combat_teacher_rerank_bash_setup": 12},
                    "avg_combat_teacher_overrides_per_game": 3.5,
                }
            },
            "ignored": "x",
        }
        compact = _compact_metrics(payload)
        assert compact["avg_floor"] == 9.1
        assert compact["action_source_counts"]["nn"] == 100
        assert compact["card_reward_skip_rate_by_boss"]["slime"] == 0.25
        assert compact["combat_teacher_override_counts"]["combat_teacher_rerank_bash_setup"] == 12
        assert "ignored" not in compact


# ===========================================================================
# TestSymbolicFeaturesHead — retrieval head wired to source_knowledge.sqlite
# ===========================================================================
#
# Coverage (9 tests, see plan at
# C:/Users/Administrator/.claude/plans/async-snacking-tome.md):
#   (a) deterministic global symbol vocab build
#   (b) per-entity id/mask shapes + coverage
#   (c) <pad>/<unk> rows zeroed
#   (d) standalone head forward shapes
#   (e) zero-init baseline parity (the critical safety net)
#   (f) relic_encoder.proj [I|0] init after enabling retrieval
#   (g) partial-load preserves old encoder columns
#   (h) optimizer ownership (PPO has, combat excludes)
#   (i) combat grad accumulation picked up by PPO optimizer


def _make_dummy_ppo_state_action_tensors(vocab):
    """Build a minimal, well-shaped state/action tensor pair for PPO forward tests.

    Uses direct torch.randint on the structured state shape contract (not via
    `_make_synthetic_state` -> `build_structured_state` because that path
    returns numpy arrays and needs more conversion; a direct tensor build
    exercises the network just as well).
    """
    from rl_encoder_v2 import (
        MAX_DECK_SIZE, MAX_HAND_SIZE, MAX_RELICS, MAX_POTIONS, MAX_ENEMIES,
        MAX_ACTIONS, MAX_MAP_NODES, MAX_CARD_REWARDS, MAX_SHOP_ITEMS,
        MAX_REST_OPTIONS, SCALAR_DIM, CARD_AUX_DIM, ENEMY_AUX_DIM,
        MAP_ROUTE_DIM, SCREEN_TYPE_TO_IDX,
    )
    from relic_tags import NUM_RELIC_TAGS

    torch.manual_seed(17)  # Deterministic across calls in a single test
    B = 2
    n_screens = len(SCREEN_TYPE_TO_IDX)

    def tl(shape, low=2, high=50):
        return torch.randint(low, high, shape, dtype=torch.long)

    state = {
        "scalars":           torch.rand(B, SCALAR_DIM),
        "deck_ids":          tl((B, MAX_DECK_SIZE)),
        "deck_aux":          torch.rand(B, MAX_DECK_SIZE, CARD_AUX_DIM),
        "deck_mask":         torch.ones(B, MAX_DECK_SIZE, dtype=torch.bool),
        "hand_ids":          tl((B, MAX_HAND_SIZE)),
        "hand_aux":          torch.rand(B, MAX_HAND_SIZE, CARD_AUX_DIM),
        "hand_mask":         torch.ones(B, MAX_HAND_SIZE, dtype=torch.bool),
        "relic_ids":         tl((B, MAX_RELICS)),
        "relic_aux":         torch.rand(B, MAX_RELICS, NUM_RELIC_TAGS),
        "relic_mask":        torch.ones(B, MAX_RELICS, dtype=torch.bool),
        "potion_ids":        tl((B, MAX_POTIONS), low=2, high=30),
        "potion_mask":       torch.ones(B, MAX_POTIONS, dtype=torch.bool),
        "enemy_ids":         tl((B, MAX_ENEMIES), low=2, high=30),
        "enemy_aux":         torch.rand(B, MAX_ENEMIES, ENEMY_AUX_DIM),
        "enemy_mask":        torch.ones(B, MAX_ENEMIES, dtype=torch.bool),
        "next_boss_idx":     tl((B,), low=0, high=10),
        "screen_type_idx":   tl((B,), low=0, high=n_screens),
        "map_node_types":    tl((B, MAX_MAP_NODES), low=0, high=5),
        "map_node_mask":     torch.ones(B, MAX_MAP_NODES, dtype=torch.bool),
        "map_route_features":torch.rand(B, MAX_MAP_NODES, MAP_ROUTE_DIM),
        "reward_card_ids":   tl((B, MAX_CARD_REWARDS)),
        "reward_card_aux":   torch.rand(B, MAX_CARD_REWARDS, CARD_AUX_DIM),
        "reward_card_mask":  torch.ones(B, MAX_CARD_REWARDS, dtype=torch.bool),
        "shop_card_ids":     tl((B, MAX_SHOP_ITEMS)),
        "shop_relic_ids":    torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.long),
        "shop_potion_ids":   torch.zeros(B, MAX_SHOP_ITEMS, dtype=torch.long),
        "shop_prices":       torch.rand(B, MAX_SHOP_ITEMS),
        "shop_mask":         torch.ones(B, MAX_SHOP_ITEMS, dtype=torch.bool),
        "rest_option_ids":   tl((B, MAX_REST_OPTIONS), low=0, high=5),
        "rest_option_mask":  torch.ones(B, MAX_REST_OPTIONS, dtype=torch.bool),
        "event_option_count":tl((B,), low=0, high=5),
    }
    action = {
        "action_type_ids":   tl((B, MAX_ACTIONS), low=0, high=10),
        "target_card_ids":   tl((B, MAX_ACTIONS), low=0, high=5),
        "target_node_types": tl((B, MAX_ACTIONS), low=0, high=5),
        "target_enemy_ids":  tl((B, MAX_ACTIONS), low=0, high=5),
        "target_indices":    tl((B, MAX_ACTIONS), low=0, high=10),
        "action_mask":       torch.ones(B, MAX_ACTIONS, dtype=torch.bool),
    }
    return state, action


class TestSymbolicFeaturesHead:

    # (a) -------------------------------------------------------------------
    def test_build_global_symbol_vocab_deterministic(self):
        from source_knowledge_features import build_global_symbol_vocab
        vocab1, sha1 = build_global_symbol_vocab()
        vocab2, sha2 = build_global_symbol_vocab()
        assert vocab1 == vocab2, "global symbol vocab is not deterministic across runs"
        assert sha1 == sha2, "sqlite SHA changed mid-test"
        assert vocab1[0] == "<pad>"
        assert vocab1[1] == "<unk>"
        # From the measurement in the plan: 399 unique + 2 specials = 401.
        # Allow a small window for DB regeneration adding/removing a handful.
        assert 380 <= len(vocab1) <= 420, f"vocab size {len(vocab1)} outside expected range"
        # Basic symbol sanity: 'Apply' is a very common command, should be present.
        assert "Apply" in vocab1

    # (b) -------------------------------------------------------------------
    def test_per_entity_symbol_id_shapes_and_coverage(self, vocab):
        from source_knowledge_features import build_all_symbol_tables
        tables, meta = build_all_symbol_tables(vocab)
        assert set(tables.keys()) == {"card", "relic", "monster", "potion"}

        card_ids, card_mask = tables["card"]
        assert card_ids.shape == (vocab.card_vocab_size, meta.card_max_len)
        assert card_mask.shape == card_ids.shape
        assert card_ids.dtype.kind == "i"
        assert card_mask.dtype == bool

        # Coverage: cards should have close to 100% non-special rows with symbols
        # (in the plan measurement it was 100.0%). Allow a small floor to be
        # robust to future DB tweaks.
        assert meta.card_coverage >= 0.95, \
            f"card symbol coverage too low: {meta.card_coverage:.1%}"
        # Monsters were ~95% in the plan measurement.
        assert meta.monster_coverage >= 0.85
        # Relics and potions have many symbol-less entries (e.g. pure stat relics
        # like amethyst_aubergine), so we only require >50%.
        assert meta.relic_coverage >= 0.50
        assert meta.potion_coverage >= 0.50

    # (c) -------------------------------------------------------------------
    def test_pad_unk_rows_zeroed(self, vocab):
        from source_knowledge_features import build_all_symbol_tables
        tables, _ = build_all_symbol_tables(vocab)
        for name in ("card", "relic", "monster", "potion"):
            ids, mask = tables[name]
            # idx 0 = <pad>, idx 1 = <unk>
            assert (ids[0] == 0).all(), f"{name} <pad> row has non-zero ids"
            assert (ids[1] == 0).all(), f"{name} <unk> row has non-zero ids"
            assert not mask[0].any(), f"{name} <pad> row has True mask bits"
            assert not mask[1].any(), f"{name} <unk> row has True mask bits"

    # (d) -------------------------------------------------------------------
    def test_symbolic_head_forward_shapes(self, vocab):
        from symbolic_features_head import SymbolicFeaturesHead
        head = SymbolicFeaturesHead(vocab, embed_dim=32, proj_dim=16)

        # Single batch, 4 slots. Include a pad (0) and a real id to verify
        # pad row is zeroed by the fully-masked guard in _attend.
        ids = torch.tensor([[5, 10, 1, 0]])  # real, real, <unk>, <pad>
        query = torch.randn(1, 4, 32)
        out = head.card(ids, query)
        assert out.shape == (1, 4, 16)
        # Because out_proj is zero-init, ALL outputs are zero at construction,
        # including the pad/unk rows.
        assert out.abs().max().item() == 0.0, \
            "SymbolicFeaturesHead output should be zero at init (zero-init out_proj)"

        # Relic / monster / potion forwards compile and produce correct shapes.
        assert head.relic(torch.tensor([[2, 5]]), torch.randn(1, 2, 32)).shape == (1, 2, 16)
        assert head.monster(torch.tensor([[3, 20]]), torch.randn(1, 2, 32)).shape == (1, 2, 16)
        assert head.potion(torch.tensor([[2]]), torch.randn(1, 1, 32)).shape == (1, 1, 16)

        # Trainable param count: should be ~18K (see plan §5).
        n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        assert 15_000 <= n_trainable <= 25_000, \
            f"unexpected trainable param count {n_trainable}"

    # (e) -------------------------------------------------------------------
    def test_symbolic_head_zero_init_baseline_parity(self, vocab):
        """Critical safety net: loading a baseline (retrieval-off) checkpoint
        into a retrieval-on model must produce bit-identical forward outputs.

        Invariant chain:
          1. SymbolicFeaturesHead.out_proj is zero-init, so head contribution = 0
          2. relic_encoder.proj is [I | 0]-init via _repair_expanded_projs, so
             baseline relic pathway passes through unchanged
          3. All other encoders are already nn.Linear, _safe_load_state_dict's
             partial copy preserves old columns bit-for-bit and zero-inits new
             columns. Since the new-column input is 0, they contribute 0.
        """
        from rl_policy_v2 import FullRunPolicyNetworkV2

        net_off = FullRunPolicyNetworkV2(vocab)
        net_off.eval()
        net_on = FullRunPolicyNetworkV2(vocab, use_symbolic_features=True, symbolic_proj_dim=16)
        net_on.eval()

        # Mimic train_hybrid._safe_load_state_dict: partial copy for wider Linears
        baseline_sd = net_off.state_dict()
        current_sd = net_on.state_dict()
        filtered = {}
        for k, v in baseline_sd.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                filtered[k] = v
            elif k in current_sd and v.dim() == 2 and current_sd[k].dim() == 2:
                if current_sd[k].shape[0] == v.shape[0] and current_sd[k].shape[1] > v.shape[1]:
                    new_w = torch.zeros_like(current_sd[k])
                    new_w[:, :v.shape[1]] = v
                    filtered[k] = new_w
        net_on.load_state_dict(filtered, strict=False)

        state, action = _make_dummy_ppo_state_action_tensors(vocab)
        with torch.no_grad():
            logits_off, values_off, dq_off, br_off, adv_off = net_off(state, action)
            logits_on,  values_on,  dq_on,  br_on,  adv_on  = net_on(state, action)

        eps = 1e-4  # float32 numerical noise floor
        assert (logits_off - logits_on).abs().max().item() < eps
        assert (values_off - values_on).abs().max().item() < eps
        assert (dq_off - dq_on).abs().max().item() < eps
        assert (br_off - br_on).abs().max().item() < eps
        assert (adv_off - adv_on).abs().max().item() < eps

    # (f) -------------------------------------------------------------------
    def test_relic_encoder_proj_identity_plus_zero(self, vocab):
        """After enabling retrieval, relic_encoder.proj should be Linear(80, 64)
        with [I | 0] initialization so baseline behavior is preserved.
        """
        from rl_policy_v2 import FullRunPolicyNetworkV2
        import torch.nn as nn

        net = FullRunPolicyNetworkV2(vocab, use_symbolic_features=True, symbolic_proj_dim=16)
        proj = net.relic_encoder.proj
        assert isinstance(proj, nn.Linear), \
            f"relic_encoder.proj is {type(proj).__name__}, expected nn.Linear"
        assert proj.in_features == 80
        assert proj.out_features == 64

        w = proj.weight
        # First 64 cols = identity
        assert torch.allclose(w[:, :64], torch.eye(64)), \
            "relic_encoder.proj first 64 cols are not identity"
        # Last 16 cols = zero
        assert w[:, 64:].abs().max().item() == 0.0, \
            "relic_encoder.proj last 16 cols are not zero"
        assert proj.bias.abs().max().item() == 0.0, \
            "relic_encoder.proj bias is not zero"

    # (g) -------------------------------------------------------------------
    def test_partial_load_preserves_old_encoder_cols(self, vocab):
        """When partial-copying a baseline deck_encoder.proj (85 cols) into a
        retrieval-on deck_encoder.proj (101 cols), the first 85 cols should be
        bit-identical and the last 16 cols zero.
        """
        from rl_policy_v2 import FullRunPolicyNetworkV2

        net_off = FullRunPolicyNetworkV2(vocab)
        baseline_weight = net_off.deck_encoder.proj.weight.detach().clone()
        baseline_bias = net_off.deck_encoder.proj.bias.detach().clone()

        net_on = FullRunPolicyNetworkV2(vocab, use_symbolic_features=True, symbolic_proj_dim=16)
        assert net_on.deck_encoder.proj.in_features == 101
        assert net_on.deck_encoder.proj.out_features == 64

        # Apply the same partial-copy logic used by train_hybrid._safe_load_state_dict
        w_old = baseline_weight
        w_new = torch.zeros_like(net_on.deck_encoder.proj.weight)
        w_new[:, :w_old.shape[1]] = w_old
        with torch.no_grad():
            net_on.deck_encoder.proj.weight.copy_(w_new)
            net_on.deck_encoder.proj.bias.copy_(baseline_bias)

        assert torch.equal(net_on.deck_encoder.proj.weight[:, :85], baseline_weight)
        assert net_on.deck_encoder.proj.weight[:, 85:].abs().max().item() == 0.0
        assert torch.equal(net_on.deck_encoder.proj.bias, baseline_bias)

    # (h) -------------------------------------------------------------------
    def test_optimizer_param_ownership(self, vocab):
        """symbolic_head.* params should be owned by the PPO optimizer and
        EXCLUDED from the combat optimizer via name-prefix filter. This avoids
        two Adam optimizer states updating the same parameter with independent
        moving averages (which would cause weight thrashing).
        """
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from combat_nn import CombatPolicyValueNetwork

        ppo = FullRunPolicyNetworkV2(vocab, use_symbolic_features=True, symbolic_proj_dim=16)
        combat = CombatPolicyValueNetwork(
            vocab=vocab, deck_repr_dim=64,
            entity_embeddings=ppo.entity_emb,
            symbolic_head=ppo.symbolic_head,
        )

        assert combat.symbolic_head is ppo.symbolic_head, \
            "symbolic_head instance is not shared between PPO and combat brains"

        ppo_opt = torch.optim.Adam(ppo.parameters(), lr=1e-4)
        combat_trainable = [
            p for n, p in combat.named_parameters()
            if not n.startswith("symbolic_head.")
        ]
        combat_opt = torch.optim.Adam(combat_trainable, lr=1e-4)

        # Build the set of param ids owned by each optimizer.
        ppo_param_ids = {id(p) for group in ppo_opt.param_groups for p in group["params"]}
        combat_param_ids = {id(p) for group in combat_opt.param_groups for p in group["params"]}
        symbolic_ids = {id(p) for p in ppo.symbolic_head.parameters()}

        # PPO owns all symbolic head params.
        assert symbolic_ids.issubset(ppo_param_ids), \
            "PPO optimizer is missing some symbolic_head params"
        # Combat owns NONE of them.
        assert symbolic_ids.isdisjoint(combat_param_ids), \
            "combat optimizer still contains symbolic_head params"
        # Sanity: combat DOES still own entity_emb (the filter is symbolic-specific,
        # not an accidental mass exclusion).
        entity_emb_ids = {id(p) for p in combat.entity_emb.parameters()}
        assert entity_emb_ids.issubset(combat_param_ids), \
            "combat optimizer lost entity_emb params"

    # (i) -------------------------------------------------------------------
    def test_combat_grad_accumulation_picked_up_by_ppo(self, vocab):
        """Gradients from combat backward must land on shared symbolic_head
        params and be stepped by the PPO optimizer.

        We construct shared PPO+combat, run a combat forward+backward to write
        a nonzero grad on `symbolic_head.symbol_embed.weight`, then step PPO's
        optimizer and verify the parameter changed. This is the autograd-level
        wiring the optimizer-ownership design depends on.
        """
        from rl_policy_v2 import FullRunPolicyNetworkV2
        from combat_nn import CombatPolicyValueNetwork
        from rl_encoder_v2 import MAX_HAND_SIZE, MAX_ENEMIES, CARD_AUX_DIM, ENEMY_AUX_DIM

        ppo = FullRunPolicyNetworkV2(vocab, use_symbolic_features=True, symbolic_proj_dim=16)
        combat = CombatPolicyValueNetwork(
            vocab=vocab, deck_repr_dim=0, pile_specific=False,
            entity_embeddings=ppo.entity_emb,
            symbolic_head=ppo.symbolic_head,
        )
        # Unfreeze the projection so it gets a real gradient. (query_proj is
        # NOT zero-init, so grads flow through it even when out_proj is zero.)
        # symbol_embed has padding_idx=0 so idx 0 grads are zeroed automatically.

        # Minimal combat state (scalars + hand + enemy only, no deck/piles).
        # The network concatenates `scalars` (legacy 18-dim) with hand/enemy
        # reprs and then appends `extra_scalars` (14-dim v2 player powers). We
        # must split these or the encoder input width doesn't match.
        B = 2
        from combat_nn import COMBAT_SCALAR_DIM, COMBAT_EXTRA_SCALAR_DIM
        state = {
            "scalars":       torch.rand(B, COMBAT_SCALAR_DIM),
            "extra_scalars": torch.rand(B, COMBAT_EXTRA_SCALAR_DIM),
            "hand_ids":   torch.randint(2, 50, (B, MAX_HAND_SIZE), dtype=torch.long),
            "hand_aux":   torch.rand(B, MAX_HAND_SIZE, CARD_AUX_DIM),
            "hand_mask":  torch.ones(B, MAX_HAND_SIZE, dtype=torch.bool),
            "enemy_ids":  torch.randint(2, 30, (B, MAX_ENEMIES), dtype=torch.long),
            "enemy_aux":  torch.rand(B, MAX_ENEMIES, ENEMY_AUX_DIM),
            "enemy_mask": torch.ones(B, MAX_ENEMIES, dtype=torch.bool),
        }
        from rl_encoder_v2 import MAX_ACTIONS
        action = {
            "action_type_ids":  torch.randint(0, 5, (B, MAX_ACTIONS), dtype=torch.long),
            "target_card_ids":  torch.randint(0, 5, (B, MAX_ACTIONS), dtype=torch.long),
            "target_enemy_ids": torch.randint(0, 5, (B, MAX_ACTIONS), dtype=torch.long),
            "action_mask":      torch.ones(B, MAX_ACTIONS, dtype=torch.bool),
        }

        # Forward+backward through combat. Use a simple loss = values.sum()
        # to create a scalar we can backprop.
        logits, values = combat(state, action)
        loss = values.sum() + logits.sum() * 0.01
        loss.backward()

        # Because SymbolicFeaturesHead.out_proj is zero-init, gradients CANNOT
        # flow backward through it to query_proj / symbol_embed / cross_attn
        # on the VERY FIRST backward (d(loss)/d(attn_out) = d(loss)/d(out) *
        # out_proj.weight.T = 0). Only out_proj.weight and out_proj.bias
        # themselves see gradient on step 1. After that first step pulls
        # out_proj away from zero, subsequent backwards flow normally to the
        # upstream params. So the right invariant is: out_proj.weight.grad is
        # nonzero on the first combat backward.
        op_grad = ppo.symbolic_head.out_proj.weight.grad
        assert op_grad is not None, \
            "symbolic_head.out_proj.weight has no grad after combat backward"
        assert op_grad.abs().max().item() > 0.0, \
            "combat backward did not accumulate grad on shared symbolic_head.out_proj"

        # Now run the PPO-owned optimizer step and verify the param moved.
        ppo_opt = torch.optim.Adam(ppo.parameters(), lr=1e-2)
        w_before = ppo.symbolic_head.out_proj.weight.detach().clone()
        ppo_opt.step()
        w_after = ppo.symbolic_head.out_proj.weight.detach().clone()
        assert not torch.equal(w_before, w_after), \
            "PPO optimizer did not update symbolic_head.out_proj.weight " \
            "despite nonzero grad — param ownership is broken"

        # Second forward-backward: now out_proj is nonzero, so gradients
        # should flow to query_proj too. This verifies the full end-to-end
        # gradient path is correct once the zero-init bootstrap phase is past.
        ppo_opt.zero_grad()
        logits2, values2 = combat(state, action)
        loss2 = values2.sum() + logits2.sum() * 0.01
        loss2.backward()
        qp_grad = ppo.symbolic_head.query_proj.weight.grad
        assert qp_grad is not None and qp_grad.abs().max().item() > 0.0, \
            "After out_proj is nonzero, gradients should flow to query_proj " \
            "through cross-attention — the autograd chain is broken"
