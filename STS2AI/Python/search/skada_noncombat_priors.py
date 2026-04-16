from __future__ import annotations

import _path_init  # noqa: F401

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from core.rl_reward_shaping import _extract_player, _extract_progress, _lower, _safe_int
from data.skada.skada_context import STARTER_DECKS, act_from_floor, is_basic_slug, is_curse_slug, slugify
from data.skada.skada_priors import SkadaPriors
from data.skada.train_campfire_model import CampfireChoiceModel
from data.skada.train_shop_core_model import ACTION_LABELS, ShopActionModel, ShopItemChoiceModel
from constants import ARTIFACTS_ROOT, DATASETS_ROOT

logger = logging.getLogger(__name__)

_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "skada"
_DEFAULT_CAMPFIRE_MODEL = _ARTIFACTS_ROOT / "ironclad_act1_relaxed_campfire_model" / "campfire_choice_best.pt"
_DEFAULT_SHOP_ACTION_MODEL = _ARTIFACTS_ROOT / "ironclad_act1_quality_shop_core_model" / "shop_action_best.pt"
_DEFAULT_SHOP_ITEM_MODEL = _ARTIFACTS_ROOT / "ironclad_act1_relaxed_shop_core_model" / "shop_item_choice_best.pt"
_DEFAULT_DB_PATH = DATASETS_ROOT / "skada" / "skada_analytics.sqlite"

_ENV_CAMPFIRE_MODEL = "STS2_SKADA_IRONCLAD_CAMPFIRE_MODEL"
_ENV_SHOP_ACTION_MODEL = "STS2_SKADA_IRONCLAD_SHOP_ACTION_MODEL"
_ENV_SHOP_ITEM_MODEL = "STS2_SKADA_IRONCLAD_SHOP_ITEM_MODEL"
_ENV_SKADA_DB = "STS2_SKADA_DB"


def _path_from_env(env_key: str, default_path: Path) -> Path:
    raw = os.environ.get(env_key, "").strip()
    return Path(raw) if raw else default_path


def _extract_character(state: dict[str, Any]) -> str:
    player = _extract_player(state)
    run = state.get("run") if isinstance(state.get("run"), dict) else {}
    for value in (
        player.get("character"),
        player.get("character_class"),
        player.get("class"),
        state.get("character"),
        run.get("character"),
    ):
        text = str(value or "").strip().upper()
        if text:
            return text
    return "UNKNOWN"


def _extract_relic_ids(player: dict[str, Any]) -> list[str]:
    relics = player.get("relics")
    if not isinstance(relics, list):
        return ["starter_relic"]
    values: list[str] = []
    for relic in relics:
        if isinstance(relic, dict):
            relic_id = str(relic.get("id") or relic.get("relic_id") or relic.get("name") or "").strip().lower()
        else:
            relic_id = str(relic or "").strip().lower()
        if relic_id:
            values.append(relic_id)
    return values or ["starter_relic"]


def _extract_deck_stats(state: dict[str, Any], character: str) -> dict[str, Any]:
    player = _extract_player(state)
    deck = player.get("deck")
    deck_cards = deck if isinstance(deck, list) else []
    deck_counts: Counter[str] = Counter()
    upgraded_counts: Counter[str] = Counter()
    canonical_ids: dict[str, str] = {}
    history: list[str] = []

    for card in deck_cards:
        if not isinstance(card, dict):
            continue
        raw_id = str(card.get("id") or card.get("card_id") or card.get("name") or "").strip()
        if not raw_id:
            continue
        upgraded = bool(card.get("is_upgraded") or card.get("upgraded") or raw_id.endswith("+"))
        canonical = raw_id.rstrip("+")
        slug = slugify(canonical)
        if not slug:
            continue
        deck_counts[slug] += 1
        canonical_ids[slug] = canonical
        history.append(slug)
        if upgraded:
            upgraded_counts[slug] += 1

    starter_slugs = {slugify(card_id) for card_id, _ in STARTER_DECKS.get(character, [])}
    starter_total = sum(count for _, count in STARTER_DECKS.get(character, []))
    starter_remaining = int(sum(deck_counts.get(slug, 0) for slug in starter_slugs))
    deck_size = int(sum(deck_counts.values()))
    upgraded_cards = int(sum(upgraded_counts.values()))

    return {
        "deck_counts": deck_counts,
        "upgraded_counts": upgraded_counts,
        "canonical_ids": canonical_ids,
        "history": history,
        "starter_slugs": starter_slugs,
        "starter_total": starter_total,
        "starter_remaining": starter_remaining,
        "deck_size": deck_size,
        "unique_cards": len([slug for slug, count in deck_counts.items() if count > 0]),
        "upgraded_cards": upgraded_cards,
        "basic_card_count": int(sum(count for slug, count in deck_counts.items() if is_basic_slug(slug))),
        "curse_card_count": int(sum(count for slug, count in deck_counts.items() if is_curse_slug(slug))),
    }


def _upgradeable_options(deck_stats: dict[str, Any], priors: SkadaPriors, floor: int) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    history = deck_stats["history"]
    for slug, count in sorted(deck_stats["deck_counts"].items()):
        unupgraded = int(count) - int(deck_stats["upgraded_counts"].get(slug, 0))
        if unupgraded <= 0:
            continue
        options.append({
            "card_id": deck_stats["canonical_ids"].get(slug, slug.upper()),
            "count_in_deck": int(count),
            "unupgraded_count": int(unupgraded),
            "context_score": priors.card_score_for_context(slug, floor, history),
        })
    options.sort(key=lambda item: (-float(item["context_score"]), str(item["card_id"])))
    return options


def _shared_feature_inputs(state: dict[str, Any], priors: SkadaPriors) -> dict[str, Any]:
    player = _extract_player(state)
    progress = _extract_progress(state)
    character = _extract_character(state)
    deck_stats = _extract_deck_stats(state, character)
    floor = max(1, _safe_int(progress.get("floor"), 1))
    max_hp = max(1.0, float(player.get("max_hp") or 80.0))
    hp_before = float(player.get("hp", player.get("current_hp", 0)) or 0.0)
    gold_before = float(player.get("gold", progress.get("gold", 0)) or 0.0)
    history = deck_stats["history"]
    upgradeable = _upgradeable_options(deck_stats, priors, floor)
    avg_history_score = (
        sum(priors.card_score_for_context(card_slug, floor, history) for card_slug in history) / len(history)
        if history else 0.5
    )
    best_upgrade_score = max((float(item["context_score"]) for item in upgradeable), default=0.5)
    mean_upgrade_score = (
        sum(float(item["context_score"]) for item in upgradeable) / len(upgradeable)
        if upgradeable else 0.5
    )
    prior_card_gains = max(0, deck_stats["deck_size"] - deck_stats["starter_total"])
    prior_shop_removes = max(0, deck_stats["starter_total"] - deck_stats["starter_remaining"])

    return {
        "character": character,
        "floor": floor,
        "act": _safe_int(progress.get("act"), act_from_floor(floor)),
        "ascension": _safe_int(progress.get("ascension", state.get("ascension")), 0),
        "hp_before": hp_before,
        "hp_ratio": hp_before / max(max_hp, 1.0),
        "gold_before": gold_before,
        "prior_card_count": deck_stats["deck_size"],
        "prior_relic_count": max(1, len(_extract_relic_ids(player))),
        "prior_card_gains": prior_card_gains,
        "prior_shop_removes": prior_shop_removes,
        "prior_upgrades": deck_stats["upgraded_cards"],
        "deck_size": deck_stats["deck_size"],
        "unique_cards": deck_stats["unique_cards"],
        "upgraded_cards": deck_stats["upgraded_cards"],
        "upgradeable_count": len(upgradeable),
        "basic_card_count": deck_stats["basic_card_count"],
        "curse_card_count": deck_stats["curse_card_count"],
        "avg_history_context_score": avg_history_score,
        "best_upgrade_context_score": best_upgrade_score,
        "mean_upgrade_context_score": mean_upgrade_score,
        "recent_damage_taken": 0.0,
        "recent_damage_dealt": 0.0,
        "recent_shop_visits": 0.0,
        "recent_elites": 0.0,
        "rest_value_proxy": max(0.0, min(1.0, 1.0 - min(1.0, hp_before / 80.0))),
        "deck_stats": deck_stats,
    }


def _campfire_feature_vector(shared: dict[str, Any]) -> list[float]:
    return [
        float(shared["floor"]) / 55.0,
        float(shared["act"]) / 3.0,
        float(shared["ascension"]) / 20.0,
        min(1.0, float(shared["hp_before"]) / 120.0),
        min(1.0, float(shared["gold_before"]) / 500.0),
        min(1.0, float(shared["prior_card_count"]) / 40.0),
        min(1.0, float(shared["prior_relic_count"]) / 20.0),
        min(1.0, float(shared["prior_card_gains"]) / 25.0),
        min(1.0, float(shared["prior_shop_removes"]) / 8.0),
        min(1.0, float(shared["prior_upgrades"]) / 12.0),
        min(1.0, float(shared["deck_size"]) / 45.0),
        min(1.0, float(shared["unique_cards"]) / 30.0),
        min(1.0, float(shared["upgraded_cards"]) / 12.0),
        min(1.0, float(shared["upgradeable_count"]) / 20.0),
        min(1.0, float(shared["basic_card_count"]) / 12.0),
        min(1.0, float(shared["curse_card_count"]) / 5.0),
        float(shared["avg_history_context_score"]),
        float(shared["best_upgrade_context_score"]),
        float(shared["mean_upgrade_context_score"]),
        0.0,
        0.0,
        0.0,
        0.0,
        float(shared["rest_value_proxy"]),
    ]


def _shop_shared_feature_vector(
    shared: dict[str, Any],
    *,
    card_option_count: int,
    relic_option_count: int,
    action_count: int,
) -> list[float]:
    return [
        float(shared["floor"]) / 55.0,
        float(shared["act"]) / 3.0,
        float(shared["ascension"]) / 20.0,
        min(1.0, float(shared["hp_before"]) / 120.0),
        min(1.0, float(shared["gold_before"]) / 500.0),
        min(1.0, float(shared["prior_card_count"]) / 40.0),
        min(1.0, float(shared["prior_relic_count"]) / 20.0),
        min(1.0, float(shared["prior_card_gains"]) / 25.0),
        min(1.0, float(shared["prior_shop_removes"]) / 8.0),
        min(1.0, float(shared["prior_upgrades"]) / 12.0),
        min(1.0, float(shared["deck_size"]) / 45.0),
        min(1.0, float(shared["unique_cards"]) / 30.0),
        min(1.0, float(shared["upgraded_cards"]) / 12.0),
        min(1.0, float(shared["upgradeable_count"]) / 20.0),
        min(1.0, float(shared["basic_card_count"]) / 12.0),
        min(1.0, float(shared["curse_card_count"]) / 5.0),
        float(shared["avg_history_context_score"]),
        float(shared["best_upgrade_context_score"]),
        float(shared["mean_upgrade_context_score"]),
        0.0,
        0.0,
        0.0,
        0.0,
        min(1.0, float(card_option_count) / 10.0),
        min(1.0, float(relic_option_count) / 5.0),
        min(1.0, float(action_count) / 6.0),
    ]


def _item_family(item: dict[str, Any] | None) -> str:
    category = _lower((item or {}).get("category") or (item or {}).get("type") or "")
    if category in {"card", "cards"}:
        return "buy_card"
    if category in {"relic"}:
        return "buy_relic"
    if category in {"card_removal", "remove_card", "remove", "purge"}:
        return "remove"
    if category in {"potion"}:
        return "buy_potion"
    return "none"


def _campfire_label_for_action(action: dict[str, Any]) -> str:
    joined = " ".join(
        _lower(action.get(key))
        for key in ("label", "type", "id", "rest_option", "option_id", "name")
    ).strip()
    if "smith" in joined or "upgrade" in joined:
        return "SMITH"
    if "rest" in joined or "heal" in joined:
        return "HEAL"
    if "hatch" in joined:
        return "HATCH"
    if "dig" in joined:
        return "DIG"
    if "lift" in joined:
        return "LIFT"
    if "toke" in joined or "purge" in joined or "remove" in joined:
        return "TOKE"
    return joined.replace(" ", "_").upper()


class IroncladSkadaNoncombatPriors:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.priors = SkadaPriors(_path_from_env(_ENV_SKADA_DB, _DEFAULT_DB_PATH))
        self.campfire_model: CampfireChoiceModel | None = None
        self.campfire_labels: list[str] = []
        self.campfire_char_vocab: dict[str, int] = {}
        self.shop_action_model: ShopActionModel | None = None
        self.shop_action_char_vocab: dict[str, int] = {}
        self.shop_item_model: ShopItemChoiceModel | None = None
        self.shop_item_vocabs: dict[str, dict[str, int]] = {}
        self.loaded = False
        self._load_models()

    def _load_models(self) -> None:
        campfire_path = _path_from_env(_ENV_CAMPFIRE_MODEL, _DEFAULT_CAMPFIRE_MODEL)
        shop_action_path = _path_from_env(_ENV_SHOP_ACTION_MODEL, _DEFAULT_SHOP_ACTION_MODEL)
        shop_item_path = _path_from_env(_ENV_SHOP_ITEM_MODEL, _DEFAULT_SHOP_ITEM_MODEL)
        try:
            if campfire_path.exists():
                payload = torch.load(campfire_path, map_location=self.device)
                feature_names = list(payload.get("feature_names") or [])
                self.campfire_labels = list(payload.get("label_names") or [])
                vocabs = payload.get("vocabs") or {}
                self.campfire_char_vocab = dict(vocabs.get("character") or {})
                self.campfire_model = CampfireChoiceModel(
                    num_characters=len(self.campfire_char_vocab),
                    num_features=len(feature_names),
                    num_labels=len(self.campfire_labels),
                ).to(self.device)
                self.campfire_model.load_state_dict(payload["model_state_dict"])
                self.campfire_model.eval()
            if shop_action_path.exists():
                payload = torch.load(shop_action_path, map_location=self.device)
                action_feature_names = list(payload.get("action_feature_names") or payload.get("feature_names") or [])
                vocabs = payload.get("action_vocabs") or payload.get("vocabs") or {}
                self.shop_action_char_vocab = dict(vocabs.get("character") or {})
                self.shop_action_model = ShopActionModel(
                    num_characters=len(self.shop_action_char_vocab),
                    num_features=len(action_feature_names),
                    num_labels=len(ACTION_LABELS),
                ).to(self.device)
                self.shop_action_model.load_state_dict(payload["model_state_dict"])
                self.shop_action_model.eval()
            if shop_item_path.exists():
                payload = torch.load(shop_item_path, map_location=self.device)
                item_feature_names = list(payload.get("item_feature_names") or payload.get("feature_names") or [])
                self.shop_item_vocabs = dict(payload.get("item_vocabs") or payload.get("vocabs") or {})
                self.shop_item_model = ShopItemChoiceModel(
                    num_items=len(self.shop_item_vocabs.get("item") or {}),
                    num_characters=len(self.shop_item_vocabs.get("character") or {}),
                    num_families=len(self.shop_item_vocabs.get("family") or {}),
                    num_features=len(item_feature_names),
                ).to(self.device)
                self.shop_item_model.load_state_dict(payload["model_state_dict"])
                self.shop_item_model.eval()
            self.loaded = any(model is not None for model in (self.campfire_model, self.shop_action_model, self.shop_item_model))
        except Exception as exc:
            logger.warning("Failed to load Ironclad non-combat priors: %s", exc)
            self.loaded = False

    def _is_supported(self, state: dict[str, Any]) -> bool:
        return self.loaded and _extract_character(state) == "IRONCLAD"

    def choose_rest_action(self, state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.campfire_model is None or not self._is_supported(state):
            return None
        options = [action for action in legal if _lower(action.get("action")) == "choose_rest_option"]
        if not options:
            return None
        shared = _shared_feature_inputs(state, self.priors)
        char_idx = self.campfire_char_vocab.get("IRONCLAD", 0)
        features = torch.tensor([_campfire_feature_vector(shared)], dtype=torch.float32, device=self.device)
        char_ids = torch.tensor([char_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.campfire_model(features, char_ids).squeeze(0).cpu()
        best_action: dict[str, Any] | None = None
        best_score = -1e9
        for action in options:
            label = _campfire_label_for_action(action)
            if label not in self.campfire_labels:
                continue
            score = float(logits[self.campfire_labels.index(label)].item())
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def choose_shop_action(self, state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.shop_action_model is None or not self._is_supported(state):
            return None
        shop = state.get("shop") if isinstance(state.get("shop"), dict) else {}
        items = shop.get("items") if isinstance(shop.get("items"), list) else []
        item_by_index: dict[int, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                item_by_index[int(item.get("index", -1))] = item
            except Exception:
                continue

        purchase_actions = [action for action in legal if _lower(action.get("action")) == "shop_purchase"]
        card_actions = [action for action in purchase_actions if _item_family(item_by_index.get(_safe_int(action.get("index"), -1))) == "buy_card"]
        relic_actions = [action for action in purchase_actions if _item_family(item_by_index.get(_safe_int(action.get("index"), -1))) == "buy_relic"]
        shared = _shared_feature_inputs(state, self.priors)
        shared_vector = _shop_shared_feature_vector(
            shared,
            card_option_count=len(card_actions),
            relic_option_count=len(relic_actions),
            action_count=len(purchase_actions),
        )
        char_idx = self.shop_action_char_vocab.get("IRONCLAD", 0)
        features = torch.tensor([shared_vector], dtype=torch.float32, device=self.device)
        char_ids = torch.tensor([char_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            action_logits = self.shop_action_model(features, char_ids).squeeze(0).cpu().tolist()
        action_scores = {label: float(action_logits[idx]) for idx, label in enumerate(ACTION_LABELS)}

        total_purchase_count = len(purchase_actions)
        card_item_bonus = self._shop_item_bonus(shared, card_actions, item_by_index, "buy_card", total_purchase_count)
        relic_item_bonus = self._shop_item_bonus(shared, relic_actions, item_by_index, "buy_relic", total_purchase_count)

        best_action: dict[str, Any] | None = None
        best_score = -1e9
        for action in legal:
            action_name = _lower(action.get("action"))
            if action_name in {"proceed", "shop_exit", "skip"}:
                score = action_scores.get("none", 0.0)
            elif action_name == "shop_purchase":
                item = item_by_index.get(_safe_int(action.get("index"), -1))
                family = _item_family(item)
                score = action_scores.get(family, -1e9)
                if family == "buy_card":
                    score += card_item_bonus.get(_safe_int(action.get("index"), -1), 0.0)
                elif family == "buy_relic":
                    score += relic_item_bonus.get(_safe_int(action.get("index"), -1), 0.0)
            else:
                continue
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _shop_item_bonus(
        self,
        shared: dict[str, Any],
        actions: list[dict[str, Any]],
        item_by_index: dict[int, dict[str, Any]],
        family: str,
        total_purchase_count: int,
    ) -> dict[int, float]:
        if self.shop_item_model is None or not actions:
            return {}
        option_features: list[list[float]] = []
        item_ids: list[int] = []
        indices: list[int] = []
        shared_vector = _shop_shared_feature_vector(
            shared,
            card_option_count=(len(actions) if family == "buy_card" else 0),
            relic_option_count=(len(actions) if family == "buy_relic" else 0),
            action_count=total_purchase_count,
        )
        deck_stats = shared["deck_stats"]
        history = deck_stats["history"]
        for action in actions:
            item = item_by_index.get(_safe_int(action.get("index"), -1))
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("card_id") or item.get("relic_id") or item.get("id") or item.get("name") or "").strip()
            slug = slugify(item_id)
            indices.append(_safe_int(action.get("index"), -1))
            item_ids.append(int((self.shop_item_vocabs.get("item") or {}).get(item_id, 0)))
            if family == "buy_card":
                prior = self.priors.card(slug)
                context_score = self.priors.card_score_for_context(slug, int(shared["floor"]), history)
                deck_synergy = self.priors.deck_synergy_boost(slug, history)
                option_features.append([
                    *shared_vector,
                    min(1.0, float(deck_stats["deck_counts"].get(slug, 0)) / 8.0),
                    1.0 if is_basic_slug(slug) else 0.0,
                    1.0 if is_curse_slug(slug) else 0.0,
                    0.0,
                    float(prior.skada_score_norm if prior else 0.5),
                    float(prior.pick_rate if prior else 0.0),
                    float(prior.win_rate_delta if prior else 0.0),
                    float(prior.hold_rate if prior else 0.0),
                    float(deck_synergy),
                    float(context_score),
                ])
            elif family == "buy_relic":
                prior = self.priors.relic(slug)
                base = float(prior.win_rate_owned if prior else 0.5)
                option_features.append([
                    *shared_vector,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    base,
                    float(prior.pick_rate if prior else 0.0),
                    float(prior.win_rate_delta if prior else 0.0),
                    float(prior.hold_rate if prior else 0.0),
                    0.0,
                    base,
                ])
        if not option_features:
            return {}
        family_idx = int((self.shop_item_vocabs.get("family") or {}).get(family, 0))
        char_idx = int((self.shop_item_vocabs.get("character") or {}).get("IRONCLAD", 0))
        option_tensor = torch.tensor([option_features], dtype=torch.float32, device=self.device)
        item_tensor = torch.tensor([item_ids], dtype=torch.long, device=self.device)
        mask_tensor = torch.ones((1, len(option_features)), dtype=torch.bool, device=self.device)
        char_tensor = torch.tensor([char_idx], dtype=torch.long, device=self.device)
        family_tensor = torch.tensor([family_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.shop_item_model(item_tensor, char_tensor, family_tensor, option_tensor, mask_tensor).squeeze(0).cpu()
        centered = logits - logits.mean()
        return {indices[idx]: float(centered[idx].item()) * 0.35 for idx in range(len(indices))}

    def choose_remove_card_action(self, state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.shop_item_model is None or not self._is_supported(state):
            return None
        card_select = state.get("card_select") if isinstance(state.get("card_select"), dict) else {}
        joined = f"{_lower(card_select.get('screen_type'))} {_lower(card_select.get('prompt'))}"
        if "remove" not in joined and "purge" not in joined and "toke" not in joined:
            return None
        select_actions = [
            action for action in legal
            if _lower(action.get("action")) in {"select_card", "combat_select_card"}
        ]
        cards = card_select.get("cards") if isinstance(card_select.get("cards"), list) else []
        if not select_actions or not cards:
            return None
        shared = _shared_feature_inputs(state, self.priors)
        deck_stats = shared["deck_stats"]
        shared_vector = _shop_shared_feature_vector(shared, card_option_count=len(cards), relic_option_count=0, action_count=1)
        option_features: list[list[float]] = []
        item_ids: list[int] = []
        action_order: list[dict[str, Any]] = []
        for action in select_actions:
            idx = _safe_int(action.get("index"), -1)
            if idx < 0 or idx >= len(cards) or not isinstance(cards[idx], dict):
                continue
            card = cards[idx]
            card_id = str(card.get("id") or card.get("card_id") or card.get("name") or "").strip().rstrip("+")
            slug = slugify(card_id)
            if not slug:
                continue
            context_score = self.priors.card_score_for_context(slug, int(shared["floor"]), deck_stats["history"])
            option_features.append([
                *shared_vector,
                min(1.0, float(max(1, deck_stats["deck_counts"].get(slug, 0))) / 8.0),
                1.0 if is_basic_slug(slug) else 0.0,
                1.0 if is_curse_slug(slug) else 0.0,
                1.0 if slug in deck_stats["starter_slugs"] and not is_basic_slug(slug) else 0.0,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                float(context_score),
            ])
            item_ids.append(int((self.shop_item_vocabs.get("item") or {}).get(card_id, 0)))
            action_order.append(action)
        if not option_features:
            return None
        family_idx = int((self.shop_item_vocabs.get("family") or {}).get("remove", 0))
        char_idx = int((self.shop_item_vocabs.get("character") or {}).get("IRONCLAD", 0))
        option_tensor = torch.tensor([option_features], dtype=torch.float32, device=self.device)
        item_tensor = torch.tensor([item_ids], dtype=torch.long, device=self.device)
        mask_tensor = torch.ones((1, len(option_features)), dtype=torch.bool, device=self.device)
        char_tensor = torch.tensor([char_idx], dtype=torch.long, device=self.device)
        family_tensor = torch.tensor([family_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.shop_item_model(item_tensor, char_tensor, family_tensor, option_tensor, mask_tensor).squeeze(0).cpu()
        best_idx = int(logits.argmax().item())
        return action_order[best_idx]


_PRIOR_BUNDLE: IroncladSkadaNoncombatPriors | None = None


def get_ironclad_noncombat_priors() -> IroncladSkadaNoncombatPriors:
    global _PRIOR_BUNDLE
    if _PRIOR_BUNDLE is None:
        _PRIOR_BUNDLE = IroncladSkadaNoncombatPriors()
    return _PRIOR_BUNDLE


def choose_prior_aware_rest_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    return get_ironclad_noncombat_priors().choose_rest_action(state, legal)


def choose_prior_aware_shop_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    return get_ironclad_noncombat_priors().choose_shop_action(state, legal)


def choose_prior_aware_remove_action(state: dict[str, Any], legal: list[dict[str, Any]]) -> dict[str, Any] | None:
    return get_ironclad_noncombat_priors().choose_remove_card_action(state, legal)
