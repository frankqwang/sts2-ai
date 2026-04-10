using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.CardSelection;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Nodes.Screens.MainMenu;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Models;

namespace STS2_MCP;

public static partial class McpMod
{
    private static Dictionary<string, object?> BuildGameState()
    {
        var result = new Dictionary<string, object?>();

        if (!RunManager.Instance.IsInProgress)
        {
            result["state_type"] = "menu";
            result["message"] = "No run in progress. Player is in the main menu.";
            result["menu"] = BuildMenuState();
            return result;
        }

        var runState = RunManager.Instance.DebugOnlyGetState();
        if (runState == null)
        {
            result["state_type"] = "unknown";
            return result;
        }

        // Card selection overlays can appear on top of any room (events, rest sites, combat)
        var topOverlay = NOverlayStack.Instance?.Peek();
        var currentRoom = runState.CurrentRoom;
        if (topOverlay is NCardGridSelectionScreen cardSelectScreen)
        {
            result["state_type"] = "card_select";
            result["card_select"] = BuildCardSelectState(cardSelectScreen, runState);
        }
        else if (topOverlay is NChooseACardSelectionScreen chooseCardScreen)
        {
            result["state_type"] = "card_select";
            result["card_select"] = BuildChooseCardState(chooseCardScreen, runState);
        }
        else if (topOverlay is NChooseARelicSelection relicSelectScreen)
        {
            result["state_type"] = "relic_select";
            result["relic_select"] = BuildRelicSelectState(relicSelectScreen, runState);
        }
        else if (topOverlay is NGameOverScreen gameOverScreen)
        {
            result["state_type"] = "game_over";
            result["game_over"] = BuildGameOverState(gameOverScreen, runState);
        }
        else if (topOverlay is IOverlayScreen
                 && topOverlay is not NRewardsScreen
                 && topOverlay is not NCardRewardSelectionScreen)
        {
            // Catch-all for unhandled overlays — prevents soft-locks
            result["state_type"] = "overlay";
            result["overlay"] = BuildOverlayState((Node)topOverlay, runState);
        }
        else if (currentRoom is CombatRoom combatRoom)
        {
            if (CombatManager.Instance.IsInProgress)
            {
                // Check for in-combat hand card selection (e.g., "Select a card to exhaust")
                var playerHand = NPlayerHand.Instance;
                if (playerHand != null && playerHand.IsInCardSelection)
                {
                    result["state_type"] = "hand_select";
                    result["hand_select"] = BuildHandSelectState(playerHand, runState);
                    result["battle"] = BuildBattleState(runState, combatRoom);
                }
                else
                {
                    result["state_type"] = combatRoom.RoomType.ToString().ToLower(); // monster, elite, boss
                    result["battle"] = BuildBattleState(runState, combatRoom);
                }
            }
            else
            {
                // After combat ends, check: map open (post-rewards) > overlays > fallback
                if (NMapScreen.Instance is { IsOpen: true })
                {
                    result["state_type"] = "map";
                    result["map"] = BuildMapState(runState);
                }
                else
                {
                    var overlay = NOverlayStack.Instance?.Peek();
                    if (overlay is NCardRewardSelectionScreen cardScreen)
                    {
                        result["state_type"] = "card_reward";
                        result["card_reward"] = BuildCardRewardState(cardScreen, runState);
                    }
                    else if (overlay is NRewardsScreen rewardsScreen)
                    {
                        result["state_type"] = "combat_rewards";
                        result["rewards"] = BuildRewardsState(rewardsScreen, runState);
                    }
                    else
                    {
                        result["state_type"] = combatRoom.RoomType.ToString().ToLower();
                        result["message"] = "Combat ended. Waiting for rewards...";
                    }
                }
            }
        }
        else if (currentRoom is EventRoom eventRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "event";
                result["event"] = BuildEventState(eventRoom, runState);
            }
        }
        else if (currentRoom is MapRoom)
        {
            result["state_type"] = "map";
            result["map"] = BuildMapState(runState);
        }
        else if (currentRoom is MerchantRoom merchantRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "shop";
                result["shop"] = BuildShopState(merchantRoom, runState);
            }
        }
        else if (currentRoom is RestSiteRoom restSiteRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "rest_site";
                result["rest_site"] = BuildRestSiteState(restSiteRoom, runState);
            }
        }
        else if (currentRoom is TreasureRoom treasureRoom)
        {
            if (NMapScreen.Instance is { IsOpen: true })
            {
                result["state_type"] = "map";
                result["map"] = BuildMapState(runState);
            }
            else
            {
                result["state_type"] = "treasure";
                result["treasure"] = BuildTreasureState(treasureRoom, runState);
            }
        }
        else
        {
            result["state_type"] = "unknown";
            result["room_type"] = currentRoom?.GetType().Name;
        }

        // Common run info
        var upcomingBoss = ResolveUpcomingBossEncounter(runState);
        var runInfo = new Dictionary<string, object?>
        {
            ["act"] = runState.CurrentActIndex + 1,
            ["floor"] = runState.TotalFloor,
            ["ascension"] = runState.AscensionLevel
        };
        if (upcomingBoss != null)
        {
            string nextBossId = upcomingBoss.Id.Entry;
            string nextBossName = SafeGetText(() => upcomingBoss.Title);
            runInfo["boss_id"] = nextBossId;
            runInfo["boss_name"] = nextBossName;
            runInfo["next_boss"] = nextBossName;
            runInfo["next_boss_id"] = nextBossId;
            runInfo["next_boss_name"] = nextBossName;
            runInfo["next_boss_archetype"] = NormalizeBossToken(nextBossId);
        }
        result["run"] = runInfo;

        // Always include full player data (relics, potions, deck, etc.) on every screen
        var _player = LocalContext.GetMe(runState);
        if (_player != null)
        {
            try
            {
                result["player"] = BuildPlayerState(_player);
            }
            catch (NullReferenceException)
            {
                // Player state may be partially initialized during screen transitions
                // (e.g. combat_rewards → map). Return minimal state to avoid crashing.
                result["player"] = new Dictionary<string, object?>
                {
                    ["character"] = SafeGetText(() => _player.Character.Title),
                    ["hp"] = 0,
                    ["max_hp"] = 1,
                    ["gold"] = 0,
                    ["status"] = new List<Dictionary<string, object?>>(),
                    ["relics"] = new List<Dictionary<string, object?>>(),
                    ["potions"] = new List<Dictionary<string, object?>>(),
                };
            }
        }

        return result;
    }

    private static EncounterModel? ResolveUpcomingBossEncounter(RunState runState)
    {
        try
        {
            if (runState.Map.SecondBossMapPoint != null
                && runState.CurrentMapCoord == runState.Map.BossMapPoint.coord
                && runState.Act.SecondBossEncounter != null)
            {
                return runState.Act.SecondBossEncounter;
            }
            return runState.Act.BossEncounter;
        }
        catch
        {
            return null;
        }
    }

    private static string NormalizeBossToken(string? rawBossId)
    {
        if (string.IsNullOrWhiteSpace(rawBossId))
            return "unknown";
        return rawBossId.Trim()
            .Replace("-", "_")
            .Replace(" ", "_")
            .ToLowerInvariant();
    }

    private static Dictionary<string, object?> BuildBattleState(RunState runState, CombatRoom combatRoom)
    {
        var combatState = CombatManager.Instance.DebugOnlyGetState();
        var battle = new Dictionary<string, object?>();

        if (combatState == null)
        {
            battle["error"] = "Combat state unavailable";
            return battle;
        }

        battle["round"] = combatState.RoundNumber;
        battle["turn"] = combatState.CurrentSide.ToString().ToLower();
        battle["is_play_phase"] = CombatManager.Instance.IsPlayPhase;

        // Enemies
        var enemies = new List<Dictionary<string, object?>>();
        var entityCounts = new Dictionary<string, int>();
        foreach (var creature in combatState.Enemies)
        {
            if (creature.IsAlive)
            {
                enemies.Add(BuildEnemyState(creature, entityCounts));
            }
        }
        battle["enemies"] = enemies;

        return battle;
    }

    private static Dictionary<string, object?> BuildPlayerState(Player player)
    {
        var state = new Dictionary<string, object?>();
        var creature = player.Creature;
        var combatState = player.PlayerCombatState;

        state["character"] = SafeGetText(() => player.Character.Title);
        // creature can be null during map transitions (post-combat screen change)
        state["hp"] = creature?.CurrentHp ?? 0;
        state["max_hp"] = creature?.MaxHp ?? 1;
        state["block"] = creature?.Block ?? 0;

        if (combatState != null)
        {
            state["energy"] = combatState.Energy;
            state["max_energy"] = combatState.MaxEnergy;

            // Stars (The Regent's resource, conditionally shown)
            if (player.Character.ShouldAlwaysShowStarCounter || combatState.Stars > 0)
            {
                state["stars"] = combatState.Stars;
            }

            // Hand
            var hand = new List<Dictionary<string, object?>>();
            int cardIndex = 0;
            foreach (var card in combatState.Hand.Cards)
            {
                hand.Add(BuildCardState(card, cardIndex));
                cardIndex++;
            }
            state["hand"] = hand;

            // Pile counts
            state["draw_pile_count"] = combatState.DrawPile.Cards.Count;
            state["discard_pile_count"] = combatState.DiscardPile.Cards.Count;
            state["exhaust_pile_count"] = combatState.ExhaustPile.Cards.Count;

            // Pile contents (draw pile is shuffled to avoid leaking actual draw order)
            var drawPileList = BuildPileCardList(combatState.DrawPile.Cards, PileType.Draw);
            ShuffleList(drawPileList);
            state["draw_pile"] = drawPileList;
            state["discard_pile"] = BuildPileCardList(combatState.DiscardPile.Cards, PileType.Discard);
            state["exhaust_pile"] = BuildPileCardList(combatState.ExhaustPile.Cards, PileType.Exhaust);

            // Orbs
            if (combatState.OrbQueue.Capacity > 0)
            {
                var orbs = new List<Dictionary<string, object?>>();
                foreach (var orb in combatState.OrbQueue.Orbs)
                {
                    // Populate SmartDescription placeholders with Focus-modified values,
                    // mirroring OrbModel.HoverTips getter (OrbModel.cs:92-94)
                    string? description = SafeGetText(() =>
                    {
                        var desc = orb.SmartDescription;
                        desc.Add("energyPrefix", orb.Owner.Character.CardPool.Title);
                        desc.Add("Passive", orb.PassiveVal);
                        desc.Add("Evoke", orb.EvokeVal);
                        return desc;
                    });
                    orbs.Add(new Dictionary<string, object?>
                    {
                        ["id"] = orb.Id.Entry,
                        ["name"] = SafeGetText(() => orb.Title),
                        ["description"] = description,
                        ["passive_val"] = orb.PassiveVal,
                        ["evoke_val"] = orb.EvokeVal,
                        ["keywords"] = BuildHoverTips(orb.HoverTips)
                    });
                }
                state["orbs"] = orbs;
                state["orb_slots"] = combatState.OrbQueue.Capacity;
                state["orb_empty_slots"] = combatState.OrbQueue.Capacity - combatState.OrbQueue.Orbs.Count;
            }
        }

        state["gold"] = player.Gold;

        // Powers (status effects)
        state["status"] = creature != null ? BuildPowersState(creature) : new List<Dictionary<string, object?>>();

        // Relics
        var relics = new List<Dictionary<string, object?>>();
        foreach (var relic in player.Relics)
        {
            relics.Add(new Dictionary<string, object?>
            {
                ["id"] = relic.Id.Entry,
                ["name"] = SafeGetText(() => relic.Title),
                ["description"] = SafeGetText(() => relic.DynamicDescription),
                ["counter"] = relic.ShowCounter ? relic.DisplayAmount : null,
                ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
            });
        }
        state["relics"] = relics;

        // Potions
        var potions = new List<Dictionary<string, object?>>();
        int slotIndex = 0;
        foreach (var potion in player.PotionSlots)
        {
            if (potion != null)
            {
                potions.Add(new Dictionary<string, object?>
                {
                    ["id"] = potion.Id.Entry,
                    ["name"] = SafeGetText(() => potion.Title),
                    ["description"] = SafeGetText(() => potion.DynamicDescription),
                    ["slot"] = slotIndex,
                    ["can_use_in_combat"] = potion.Usage == PotionUsage.CombatOnly || potion.Usage == PotionUsage.AnyTime,
                    ["target_type"] = potion.TargetType.ToString(),
                    ["keywords"] = BuildHoverTips(potion.ExtraHoverTips)
                });
            }
            slotIndex++;
        }
        state["potions"] = potions;

        return state;
    }

    private static Dictionary<string, object?> BuildCardState(CardModel card, int index)
    {
        string costDisplay;
        if (card.EnergyCost.CostsX)
            costDisplay = "X";
        else
        {
            int cost = card.EnergyCost.GetAmountToSpend();
            costDisplay = cost.ToString();
        }

        card.CanPlay(out var unplayableReason, out _);

        // Star cost (The Regent's cards; CanonicalStarCost >= 0 means card has a star cost)
        string? starCostDisplay = null;
        if (card.HasStarCostX)
            starCostDisplay = "X";
        else if (card.CurrentStarCost >= 0)
            starCostDisplay = card.GetStarCostWithModifiers().ToString();

        return new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = card.Id.Entry,
            ["name"] = card.Title,
            ["type"] = card.Type.ToString(),
            ["cost"] = costDisplay,
            ["star_cost"] = starCostDisplay,
            ["description"] = SafeGetCardDescription(card),
            ["target_type"] = card.TargetType.ToString(),
            ["can_play"] = unplayableReason == UnplayableReason.None,
            ["unplayable_reason"] = unplayableReason != UnplayableReason.None ? unplayableReason.ToString() : null,
            ["is_upgraded"] = card.IsUpgraded,
            ["keywords"] = BuildHoverTips(card.HoverTips)
        };
    }

    private static void ShuffleList<T>(List<T> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = Random.Shared.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static List<Dictionary<string, object?>> BuildPileCardList(IEnumerable<CardModel> cards, PileType pile)
    {
        var list = new List<Dictionary<string, object?>>();
        foreach (var card in cards)
        {
            list.Add(new Dictionary<string, object?>
            {
                ["name"] = SafeGetText(() => card.Title),
                ["description"] = SafeGetCardDescription(card, pile)
            });
        }
        return list;
    }

    private static Dictionary<string, object?> BuildEnemyState(Creature creature, Dictionary<string, int> entityCounts)
    {
        var monster = creature.Monster;
        string baseId = monster?.Id.Entry ?? "unknown";

        // Generate entity_id like "jaw_worm_0"
        if (!entityCounts.TryGetValue(baseId, out int count))
            count = 0;
        entityCounts[baseId] = count + 1;
        string entityId = $"{baseId}_{count}";

        var state = new Dictionary<string, object?>
        {
            ["entity_id"] = entityId,
            ["combat_id"] = creature.CombatId,
            ["name"] = SafeGetText(() => monster?.Title),
            ["hp"] = creature.CurrentHp,
            ["max_hp"] = creature.MaxHp,
            ["block"] = creature.Block,
            ["is_alive"] = creature.IsAlive,
            ["is_hittable"] = creature.IsHittable,
            ["status"] = BuildPowersState(creature)
        };

        // Intents
        if (monster?.NextMove is MoveState moveState)
        {
            var intents = new List<Dictionary<string, object?>>();
            foreach (var intent in moveState.Intents)
            {
                var intentData = new Dictionary<string, object?>
                {
                    ["type"] = intent.IntentType.ToString()
                };
                try
                {
                    var targets = creature.CombatState?.PlayerCreatures;
                    if (targets != null)
                    {
                        string label = intent.GetIntentLabel(targets, creature).GetFormattedText();
                        intentData["label"] = StripRichTextTags(label);

                        var hoverTip = intent.GetHoverTip(targets, creature);
                        if (hoverTip.Title != null)
                            intentData["title"] = StripRichTextTags(hoverTip.Title);
                        if (hoverTip.Description != null)
                            intentData["description"] = StripRichTextTags(hoverTip.Description);
                    }
                }
                catch { /* intent label may fail for some types */ }
                intents.Add(intentData);
            }
            state["intents"] = intents;
        }

        return state;
    }

    private static Dictionary<string, object?> BuildNonCombatPlayerState(Player player)
    {
        int totalSlots = player.PotionSlots.Count;
        int openSlots = player.PotionSlots.Count(s => s == null);

        var state = new Dictionary<string, object?>
        {
            ["character"] = SafeGetText(() => player.Character.Title),
            ["hp"] = player.Creature.CurrentHp,
            ["max_hp"] = player.Creature.MaxHp,
            ["gold"] = player.Gold,
            ["potion_slots"] = totalSlots,
            ["open_potion_slots"] = openSlots
        };

        var deck = new List<Dictionary<string, object?>>();
        foreach (var card in player.Deck.Cards)
        {
            string costDisplay = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString();
            string? starCostDisplay = null;
            if (card.HasStarCostX)
                starCostDisplay = "X";
            else if (card.CurrentStarCost >= 0)
                starCostDisplay = card.GetStarCostWithModifiers().ToString();

            deck.Add(new Dictionary<string, object?>
            {
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = costDisplay,
                ["star_cost"] = starCostDisplay,
                ["description"] = SafeGetCardDescription(card, PileType.Deck),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
        }
        state["deck"] = deck;

        var relics = new List<Dictionary<string, object?>>();
        foreach (var relic in player.Relics)
        {
            relics.Add(new Dictionary<string, object?>
            {
                ["id"] = relic.Id.Entry,
                ["name"] = SafeGetText(() => relic.Title),
                ["description"] = SafeGetText(() => relic.DynamicDescription),
                ["counter"] = relic.ShowCounter ? relic.DisplayAmount : null,
                ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
            });
        }
        state["relics"] = relics;

        var potions = new List<Dictionary<string, object?>>();
        int slotIndex = 0;
        foreach (var potion in player.PotionSlots)
        {
            if (potion != null)
            {
                potions.Add(new Dictionary<string, object?>
                {
                    ["id"] = potion.Id.Entry,
                    ["name"] = SafeGetText(() => potion.Title),
                    ["description"] = SafeGetText(() => potion.DynamicDescription),
                    ["slot"] = slotIndex,
                    ["can_use_in_combat"] = potion.Usage == PotionUsage.CombatOnly || potion.Usage == PotionUsage.AnyTime,
                    ["target_type"] = potion.TargetType.ToString(),
                    ["keywords"] = BuildHoverTips(potion.ExtraHoverTips)
                });
            }
            slotIndex++;
        }
        state["potions"] = potions;

        return state;
    }

    private static Dictionary<string, object?> BuildEventState(EventRoom eventRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        var eventModel = eventRoom.CanonicalEvent;
        bool isAncient = eventModel is AncientEventModel;
        state["event_id"] = eventModel.Id.Entry;
        state["event_name"] = SafeGetText(() => eventModel.Title);
        state["is_ancient"] = isAncient;

        // Check dialogue state for ancients
        bool inDialogue = false;
        var uiRoom = NEventRoom.Instance;
        if (isAncient && uiRoom != null)
        {
            var ancientLayout = FindFirst<NAncientEventLayout>(uiRoom);
            if (ancientLayout != null)
            {
                var hitbox = ancientLayout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
                inDialogue = hitbox != null && hitbox.Visible && hitbox.IsEnabled;
            }
        }
        state["in_dialogue"] = inDialogue;

        // Event body text
        state["body"] = SafeGetText(() => eventModel.Description);

        // Options from UI
        var options = new List<Dictionary<string, object?>>();
        if (uiRoom != null)
        {
            var buttons = FindAll<NEventOptionButton>(uiRoom);
            int index = 0;
            foreach (var button in buttons)
            {
                var opt = button.Option;
                var optData = new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["text_key"] = opt.TextKey,
                    ["title"] = SafeGetText(() => opt.Title),
                    ["description"] = SafeGetText(() => opt.Description),
                    ["is_locked"] = opt.IsLocked,
                    ["is_proceed"] = opt.IsProceed,
                    ["was_chosen"] = opt.WasChosen
                };
                var effectInfo = InferEventOptionEffects(opt);
                foreach (var kv in effectInfo)
                    optData[kv.Key] = kv.Value;
                if (opt.Relic != null)
                {
                    optData["relic_name"] = SafeGetText(() => opt.Relic.Title);
                    optData["relic_description"] = SafeGetText(() => opt.Relic.DynamicDescription);
                }
                optData["keywords"] = BuildHoverTips(opt.HoverTips);
                options.Add(optData);
                index++;
            }
        }
        state["options"] = options;

        return state;
    }

    private static Dictionary<string, object?> BuildMenuState()
    {
        var availableActions = new List<string>();
        var state = new Dictionary<string, object?>
        {
            ["is_main_menu_visible"] = false,
            ["has_run_save"] = false,
            ["can_open_singleplayer"] = false,
            ["singleplayer_submenu_visible"] = false,
            ["character_select_visible"] = false,
            ["selected_character"] = null,
            ["ascension"] = 0,
            ["max_ascension"] = 0,
            ["can_start"] = false,
            ["available_actions"] = availableActions
        };

        try
        {
            state["has_run_save"] = SaveManager.Instance.HasRunSave;
        }
        catch
        {
            state["has_run_save"] = false;
        }

        var mainMenu = NGame.Instance?.MainMenu;
        if (mainMenu == null)
        {
            state["available_characters"] = BuildMenuCharactersFromProgress();
            return state;
        }

        state["is_main_menu_visible"] = mainMenu.IsVisibleInTree();

        var singleplayerButton = mainMenu.GetNodeOrNull<NButton>("MainMenuTextButtons/SingleplayerButton");
        state["can_open_singleplayer"] = singleplayerButton is { Visible: true, IsEnabled: true };
        if ((bool)state["can_open_singleplayer"]!)
            availableActions.Add("select_character");

        var singleplayerSubmenu = mainMenu.GetNodeOrNull<Control>("Submenus/SingleplayerSubmenu");
        state["singleplayer_submenu_visible"] = singleplayerSubmenu?.Visible ?? false;

        var charSelectScreen = mainMenu.GetNodeOrNull<NCharacterSelectScreen>("Submenus/CharacterSelectScreen");
        bool isCharSelectVisible = charSelectScreen?.Visible == true && charSelectScreen.IsVisibleInTree();
        state["character_select_visible"] = isCharSelectVisible;

        if (isCharSelectVisible && charSelectScreen != null)
        {
            state["available_characters"] = BuildMenuCharactersFromCharacterSelect(charSelectScreen);
            try
            {
                state["selected_character"] = charSelectScreen.Lobby.LocalPlayer.character.Id.Entry;
                state["ascension"] = charSelectScreen.Lobby.Ascension;
                state["max_ascension"] = charSelectScreen.Lobby.MaxAscension;
            }
            catch
            {
                state["selected_character"] = null;
                state["ascension"] = 0;
                state["max_ascension"] = 0;
            }

            var confirmButton = charSelectScreen.GetNodeOrNull<NConfirmButton>("ConfirmButton");
            state["can_start"] = confirmButton is { Visible: true, IsEnabled: true };
            availableActions.Add("select_character");
            availableActions.Add("set_ascension");
            if ((bool)state["can_start"]!)
                availableActions.Add("start_run");
        }
        else
        {
            state["available_characters"] = BuildMenuCharactersFromProgress();
        }

        return state;
    }

    private static List<Dictionary<string, object?>> BuildMenuCharactersFromCharacterSelect(NCharacterSelectScreen screen)
    {
        var buttonContainer = screen.GetNodeOrNull<Node>("CharSelectButtons/ButtonContainer");
        var buttons = buttonContainer != null ? FindAll<NCharacterSelectButton>(buttonContainer) : new List<NCharacterSelectButton>();

        string? selectedId = null;
        try
        {
            selectedId = screen.Lobby.LocalPlayer.character.Id.Entry;
        }
        catch
        {
            selectedId = null;
        }

        var characters = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var button in buttons)
        {
            characters.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = button.Character.Id.Entry,
                ["name"] = SafeGetText(() => button.Character.Title),
                ["is_locked"] = button.IsLocked,
                ["is_selected"] = selectedId != null && selectedId == button.Character.Id.Entry,
                ["is_random"] = button.IsRandom
            });
            index++;
        }

        return characters;
    }

    private static List<Dictionary<string, object?>> BuildMenuCharactersFromProgress()
    {
        var characters = new List<Dictionary<string, object?>>();

        HashSet<MegaCrit.Sts2.Core.Models.CharacterModel>? unlocked = null;
        try
        {
            unlocked = SaveManager.Instance.GenerateUnlockStateFromProgress().Characters.ToHashSet();
        }
        catch
        {
            unlocked = null;
        }

        int index = 0;
        foreach (var character in ModelDb.AllCharacters)
        {
            bool isLocked = unlocked != null && !unlocked.Contains(character);
            characters.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = character.Id.Entry,
                ["name"] = SafeGetText(() => character.Title),
                ["is_locked"] = isLocked,
                ["is_selected"] = false,
                ["is_random"] = false
            });
            index++;
        }

        bool allUnlocked = unlocked != null && ModelDb.AllCharacters.All(unlocked.Contains);
        if (allUnlocked)
        {
            var random = ModelDb.Character<RandomCharacter>();
            characters.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = random.Id.Entry,
                ["name"] = SafeGetText(() => random.Title),
                ["is_locked"] = false,
                ["is_selected"] = false,
                ["is_random"] = true
            });
        }

        return characters;
    }

    private static Dictionary<string, object?> BuildGameOverState(NGameOverScreen screen, RunState runState)
    {
        var state = BuildOverlayState(screen, runState);
        state["kind"] = "game_over";
        state["terminal"] = true;
        state["outcome"] = runState.CurrentRoom?.IsVictoryRoom == true ? "victory" : "death";
        return state;
    }

    private static Dictionary<string, object?> BuildOverlayState(Node overlay, RunState runState)
    {
        var state = new Dictionary<string, object?>
        {
            ["screen_type"] = overlay.GetType().Name,
            ["kind"] = "generic_overlay",
            ["terminal"] = false,
            ["message"] = $"An overlay ({overlay.GetType().Name}) is active. It may require manual interaction in-game."
        };

        var buttons = new List<Dictionary<string, object?>>();
        var availableActions = new List<Dictionary<string, object?>>();
        bool canConfirm = false;
        bool canCancel = false;
        int index = 0;

        foreach (var button in FindAll<NClickableControl>(overlay).Where(b => b.Visible && b.IsVisibleInTree()))
        {
            string text = GetOverlayButtonText(button) ?? button.Name;
            bool isConfirm = IsConfirmLike(text, button.Name);
            bool isCancel = IsCancelLike(text, button.Name);

            buttons.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["name"] = button.Name,
                ["text"] = text,
                ["is_enabled"] = button.IsEnabled,
                ["is_visible"] = button.Visible,
                ["is_confirm"] = isConfirm,
                ["is_cancel"] = isCancel
            });

            if (button.IsEnabled)
            {
                availableActions.Add(new Dictionary<string, object?>
                {
                    ["action"] = "overlay_press",
                    ["index"] = index,
                    ["name"] = button.Name,
                    ["text"] = text,
                    ["is_confirm"] = isConfirm,
                    ["is_cancel"] = isCancel
                });
            }

            if (isConfirm && button.IsEnabled)
                canConfirm = true;
            if (isCancel && button.IsEnabled)
                canCancel = true;

            index++;
        }

        state["buttons"] = buttons;
        state["available_actions"] = availableActions;
        state["primary_text"] = ExtractOverlayPrimaryText(overlay);
        state["can_confirm"] = canConfirm;
        state["can_cancel"] = canCancel;
        if (overlay is NGameOverScreen)
        {
            state["kind"] = "game_over";
            state["terminal"] = true;
            state["outcome"] = runState.CurrentRoom?.IsVictoryRoom == true ? "victory" : "death";
        }
        return state;
    }

    private static string? ExtractOverlayPrimaryText(Node overlay)
    {
        string? best = null;
        foreach (var node in EnumerateNodeTree(overlay))
        {
            if (node is NClickableControl)
                continue;

            string? text = TryGetNodeText(node);
            if (string.IsNullOrWhiteSpace(text) || text.Length < 3)
                continue;

            if (best == null || text.Length > best.Length)
                best = text;
        }
        return best;
    }

    private static string? GetOverlayButtonText(NClickableControl button)
    {
        string? direct = TryGetNodeText(button);
        if (!string.IsNullOrWhiteSpace(direct))
            return direct;

        foreach (var node in EnumerateNodeTree(button))
        {
            if (node == button)
                continue;
            string? text = TryGetNodeText(node);
            if (!string.IsNullOrWhiteSpace(text))
                return text;
        }
        return null;
    }

    private static IEnumerable<Node> EnumerateNodeTree(Node root)
    {
        yield return root;
        foreach (var child in root.GetChildren())
        {
            if (child is not Node node)
                continue;

            foreach (var descendant in EnumerateNodeTree(node))
                yield return descendant;
        }
    }

    private static string? TryGetNodeText(Node node)
    {
        foreach (string key in new[] { "text", "title", "label", "Text", "Title" })
        {
            try
            {
                var value = node.Get(key);
                if (value.VariantType == Variant.Type.Nil)
                    continue;

                string normalized = NormalizeText(value.AsString());
                if (!string.IsNullOrWhiteSpace(normalized))
                    return normalized;
            }
            catch
            {
                // no-op
            }
        }

        return null;
    }

    private static string NormalizeText(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        string stripped = StripRichTextTags(text).Replace("\n", " ").Replace("\r", " ");
        return Regex.Replace(stripped, "\\s+", " ").Trim();
    }

    private static bool IsConfirmLike(string? text, string? name)
    {
        string value = (text + " " + name).ToLowerInvariant();
        return value.Contains("confirm")
               || value.Contains("yes")
               || value.Contains("ok")
               || value.Contains("accept")
               || value.Contains("proceed")
               || value.Contains("continue")
               || value.Contains("start")
               || value.Contains("embark")
               || value.Contains("ready")
               || value.Contains("确认")
               || value.Contains("继续")
               || value.Contains("开始")
               || value.Contains("前进")
               || value.Contains("出发");
    }

    private static bool IsCancelLike(string? text, string? name)
    {
        string value = (text + " " + name).ToLowerInvariant();
        return value.Contains("cancel")
               || value.Contains("back")
               || value.Contains("close")
               || value.Contains("exit")
               || value.Contains("no")
               || value.Contains("abort")
               || value.Contains("dismiss")
               || value.Contains("skip")
               || value.Contains("取消")
               || value.Contains("返回")
               || value.Contains("关闭")
               || value.Contains("否")
               || value.Contains("跳过");
    }

    private static Dictionary<string, object?> InferEventOptionEffects(EventOption option)
    {
        var effectTags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var cardOps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        string combined = string.Join(" ",
            new[]
            {
                option.TextKey,
                SafeGetText(() => option.Title),
                SafeGetText(() => option.Description)
            }.Where(s => !string.IsNullOrWhiteSpace(s)));
        string normalized = NormalizeText(combined).ToLowerInvariant();

        int? hpDelta = InferHpDelta(normalized);
        int? goldDelta = InferGoldDelta(normalized);
        if (hpDelta.HasValue)
            effectTags.Add(hpDelta.Value < 0 ? "lose_hp" : "gain_hp");
        if (goldDelta.HasValue)
            effectTags.Add(goldDelta.Value < 0 ? "lose_gold" : "gain_gold");

        bool addsCurse = normalized.Contains("curse") || normalized.Contains("诅咒");
        if (addsCurse)
            effectTags.Add("add_curse");

        bool startsCombat = normalized.Contains("combat")
                            || normalized.Contains("battle")
                            || normalized.Contains("fight")
                            || normalized.Contains("enemy")
                            || normalized.Contains("战斗")
                            || normalized.Contains("遭遇");
        if (startsCombat)
            effectTags.Add("start_combat");
        if (option.IsProceed)
            effectTags.Add("proceed");

        if (normalized.Contains("remove") || normalized.Contains("purge") || normalized.Contains("移除"))
        {
            cardOps.Add("remove");
            effectTags.Add("card_remove");
        }
        if (normalized.Contains("upgrade") || normalized.Contains("smith") || normalized.Contains("升级") || normalized.Contains("强化"))
        {
            cardOps.Add("upgrade");
            effectTags.Add("card_upgrade");
        }
        if (normalized.Contains("transform") || normalized.Contains("变形") || normalized.Contains("转换"))
        {
            cardOps.Add("transform");
            effectTags.Add("card_transform");
        }
        if ((normalized.Contains("gain") || normalized.Contains("obtain") || normalized.Contains("receive") || normalized.Contains("获得") || normalized.Contains("得到"))
            && (normalized.Contains("card") || normalized.Contains("卡")))
        {
            cardOps.Add("add");
            effectTags.Add("card_add");
        }

        return new Dictionary<string, object?>
        {
            ["effect_tags"] = effectTags.OrderBy(t => t).ToList(),
            ["hp_delta"] = hpDelta,
            ["gold_delta"] = goldDelta,
            ["card_ops"] = cardOps.OrderBy(t => t).ToList(),
            ["adds_curse"] = addsCurse,
            ["starts_combat"] = startsCombat
        };
    }

    private static int? InferHpDelta(string normalized)
    {
        var match = Regex.Match(normalized, "(gain|heal|recover|回复|恢复|获得)\\s*(\\d+)\\s*(hp|health|生命|体力)");
        if (match.Success)
            return int.Parse(match.Groups[2].Value);

        match = Regex.Match(normalized, "(lose|pay|take|suffer|失去|损失|受到)\\s*(\\d+)\\s*(hp|health|生命|体力|damage|伤害)");
        if (match.Success)
            return -int.Parse(match.Groups[2].Value);

        match = Regex.Match(normalized, "([+-]\\d+)\\s*(hp|health|生命|体力)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out int signed))
            return signed;

        return null;
    }

    private static int? InferGoldDelta(string normalized)
    {
        var match = Regex.Match(normalized, "(gain|obtain|receive|获得|得到)\\s*(\\d+)\\s*(gold|金币)");
        if (match.Success)
            return int.Parse(match.Groups[2].Value);

        match = Regex.Match(normalized, "(lose|pay|spend|失去|花费)\\s*(\\d+)\\s*(gold|金币)");
        if (match.Success)
            return -int.Parse(match.Groups[2].Value);

        match = Regex.Match(normalized, "([+-]\\d+)\\s*(gold|金币)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out int signed))
            return signed;

        return null;
    }

    private static Dictionary<string, object?> BuildRestSiteState(RestSiteRoom restSiteRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        var options = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var opt in restSiteRoom.Options)
        {
            options.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = opt.OptionId,
                ["name"] = SafeGetText(() => opt.Title),
                ["description"] = SafeGetText(() => opt.Description),
                ["is_enabled"] = opt.IsEnabled
            });
            index++;
        }
        state["options"] = options;

        var proceedButton = NRestSiteRoom.Instance?.ProceedButton;
        state["can_proceed"] = proceedButton?.IsEnabled ?? false;

        return state;
    }

    private static Dictionary<string, object?> BuildShopState(MerchantRoom merchantRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        var inventory = merchantRoom.Inventory;
        var items = new List<Dictionary<string, object?>>();
        int index = 0;

        // Cards
        foreach (var entry in inventory.CardEntries)
        {
            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "card",
                ["cost"] = entry.Cost,
                ["is_stocked"] = entry.IsStocked,
                ["can_afford"] = entry.EnoughGold,
                ["on_sale"] = entry.IsOnSale
            };
            if (entry.CreationResult?.Card is { } card)
            {
                item["card_id"] = card.Id.Entry;
                item["card_name"] = SafeGetText(() => card.Title);
                item["card_type"] = card.Type.ToString();
                item["card_rarity"] = card.Rarity.ToString();
                item["card_description"] = SafeGetCardDescription(card, PileType.None);
                item["keywords"] = BuildHoverTips(card.HoverTips);
            }
            items.Add(item);
            index++;
        }

        // Relics
        foreach (var entry in inventory.RelicEntries)
        {
            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "relic",
                ["cost"] = entry.Cost,
                ["is_stocked"] = entry.IsStocked,
                ["can_afford"] = entry.EnoughGold
            };
            if (entry.Model is { } relic)
            {
                item["relic_id"] = relic.Id.Entry;
                item["relic_name"] = SafeGetText(() => relic.Title);
                item["relic_description"] = SafeGetText(() => relic.DynamicDescription);
                item["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic);
            }
            items.Add(item);
            index++;
        }

        // Potions
        foreach (var entry in inventory.PotionEntries)
        {
            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "potion",
                ["cost"] = entry.Cost,
                ["is_stocked"] = entry.IsStocked,
                ["can_afford"] = entry.EnoughGold
            };
            if (entry.Model is { } potion)
            {
                item["potion_id"] = potion.Id.Entry;
                item["potion_name"] = SafeGetText(() => potion.Title);
                item["potion_description"] = SafeGetText(() => potion.DynamicDescription);
                item["keywords"] = BuildHoverTips(potion.ExtraHoverTips);
            }
            items.Add(item);
            index++;
        }

        // Card removal
        if (inventory.CardRemovalEntry is { } removal)
        {
            items.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["category"] = "card_removal",
                ["cost"] = removal.Cost,
                ["is_stocked"] = removal.IsStocked,
                ["can_afford"] = removal.EnoughGold
            });
        }

        state["items"] = items;

        var proceedButton = NMerchantRoom.Instance?.ProceedButton;
        state["can_proceed"] = proceedButton?.IsEnabled ?? false;

        return state;
    }

    private static Dictionary<string, object?> BuildMapState(RunState runState)
    {
        var state = new Dictionary<string, object?>();

        // Player summary
        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        var map = runState.Map;
        var visitedCoords = runState.VisitedMapCoords;

        // Current position
        if (visitedCoords.Count > 0)
        {
            var cur = visitedCoords[visitedCoords.Count - 1];
            state["current_position"] = new Dictionary<string, object?>
            {
                ["col"] = cur.col, ["row"] = cur.row,
                ["type"] = map.GetPoint(cur)?.PointType.ToString()
            };
        }

        // Visited path
        var visited = new List<Dictionary<string, object?>>();
        foreach (var coord in visitedCoords)
        {
            visited.Add(new Dictionary<string, object?>
            {
                ["col"] = coord.col, ["row"] = coord.row,
                ["type"] = map.GetPoint(coord)?.PointType.ToString()
            });
        }
        state["visited"] = visited;

        // Next options — read travelable state from UI nodes
        var nextOptions = new List<Dictionary<string, object?>>();
        var mapScreen = NMapScreen.Instance;
        if (mapScreen != null)
        {
            var travelable = FindAll<NMapPoint>(mapScreen)
                .Where(mp => mp.State == MapPointState.Travelable)
                .OrderBy(mp => mp.Point.coord.col)
                .ToList();

            int index = 0;
            foreach (var nmp in travelable)
            {
                var pt = nmp.Point;
                var option = new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["col"] = pt.coord.col,
                    ["row"] = pt.coord.row,
                    ["type"] = pt.PointType.ToString()
                };

                // 1-level lookahead
                var children = pt.Children
                    .OrderBy(c => c.coord.col)
                    .Select(c => new Dictionary<string, object?>
                    {
                        ["col"] = c.coord.col, ["row"] = c.coord.row,
                        ["type"] = c.PointType.ToString()
                    }).ToList();
                if (children.Count > 0)
                    option["leads_to"] = children;

                nextOptions.Add(option);
                index++;
            }
        }
        state["next_options"] = nextOptions;

        // Full map — all nodes organized for planning
        var nodes = new List<Dictionary<string, object?>>();

        // Starting point
        var start = map.StartingMapPoint;
        nodes.Add(BuildMapNode(start));

        // Grid nodes
        foreach (var pt in map.GetAllMapPoints())
            nodes.Add(BuildMapNode(pt));

        // Boss
        nodes.Add(BuildMapNode(map.BossMapPoint));
        if (map.SecondBossMapPoint != null)
            nodes.Add(BuildMapNode(map.SecondBossMapPoint));

        state["nodes"] = nodes;
        state["boss"] = new Dictionary<string, object?>
        {
            ["col"] = map.BossMapPoint.coord.col,
            ["row"] = map.BossMapPoint.coord.row
        };

        return state;
    }

    private static Dictionary<string, object?> BuildMapNode(MapPoint pt)
    {
        return new Dictionary<string, object?>
        {
            ["col"] = pt.coord.col,
            ["row"] = pt.coord.row,
            ["type"] = pt.PointType.ToString(),
            ["children"] = pt.Children
                .OrderBy(c => c.coord.col)
                .Select(c => new List<int> { c.coord.col, c.coord.row })
                .ToList()
        };
    }

    private static Dictionary<string, object?> BuildRewardsState(NRewardsScreen rewardsScreen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        // Player summary for decision-making context
        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        state["reward_source"] = GetRewardSourceName(runState);
        // Reward items
        var rewardButtons = FindAll<NRewardButton>(rewardsScreen);
        var items = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var button in rewardButtons)
        {
            if (button.Reward == null || !button.IsEnabled) continue;
            var reward = button.Reward;

            var item = new Dictionary<string, object?>
            {
                ["index"] = index,
                ["type"] = GetRewardTypeName(reward),
                ["description"] = SafeGetText(() => reward.Description),
                ["reward_key"] = BuildRewardKey(reward),
                ["reward_source"] = GetRewardSourceName(runState)
            };

            bool claimable = IsRewardClaimable(reward, player, out string? claimBlockReason);
            item["claimable"] = claimable;
            if (!string.IsNullOrWhiteSpace(claimBlockReason))
                item["claim_block_reason"] = claimBlockReason;

            // Type-specific details
            if (reward is GoldReward goldReward)
                item["gold_amount"] = goldReward.Amount;
            else if (reward is PotionReward potionReward && potionReward.Potion != null)
            {
                item["potion_id"] = potionReward.Potion.Id.Entry;
                item["potion_name"] = SafeGetText(() => potionReward.Potion.Title);
            }

            items.Add(item);
            index++;
        }
        state["items"] = items;

        // Proceed button
        var proceedButton = FindFirst<NProceedButton>(rewardsScreen);
        state["can_proceed"] = proceedButton?.IsEnabled ?? false;

        return state;
    }

    private static Dictionary<string, object?> BuildCardRewardState(NCardRewardSelectionScreen cardScreen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);

        var cardHolders = FindAllSortedByPosition<NCardHolder>(cardScreen);
        var cards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in cardHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            string costDisplay = card.EnergyCost.CostsX
                ? "X"
                : card.EnergyCost.GetAmountToSpend().ToString();

            string? starCostDisplay = null;
            if (card.HasStarCostX)
                starCostDisplay = "X";
            else if (card.CurrentStarCost >= 0)
                starCostDisplay = card.GetStarCostWithModifiers().ToString();

            cards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = costDisplay,
                ["star_cost"] = starCostDisplay,
                ["description"] = SafeGetCardDescription(card, PileType.None),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = cards;

        var altButtons = FindAll<NCardRewardAlternativeButton>(cardScreen);
        state["can_skip"] = altButtons.Count > 0;

        return state;
    }

    private static Dictionary<string, object?> BuildCardSelectState(NCardGridSelectionScreen screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();
        var selectedCards = GetSelectedCards(screen);
        TryGetCardSelectorPrefs(screen, out var prefs);

        // Screen type
        state["screen_type"] = screen switch
        {
            NDeckTransformSelectScreen => "transform",
            NDeckUpgradeSelectScreen => "upgrade",
            NDeckCardSelectScreen => "select",
            NSimpleCardSelectScreen => "simple_select",
            _ => screen.GetType().Name
        };

        // Player summary
        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        // Prompt text from UI label
        var bottomLabel = screen.GetNodeOrNull("%BottomLabel");
        if (bottomLabel != null)
        {
            var textVariant = bottomLabel.Get("text");
            string? prompt = textVariant.VariantType != Godot.Variant.Type.Nil ? StripRichTextTags(textVariant.AsString()) : null;
            state["prompt"] = prompt;
        }

        // Cards in the grid (sorted by visual position — MoveToFront can reorder children)
        var cardHolders = FindAllSortedByPosition<NGridCardHolder>(screen);
        var cards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in cardHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            cards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["description"] = SafeGetCardDescription(card, PileType.None),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = cards;
        state["selected_cards"] = BuildCompactCardList(selectedCards);
        state["selected_count"] = selectedCards.Count;
        state["min_select"] = prefs.MinSelect;
        state["max_select"] = prefs.MaxSelect;
        state["remaining_picks"] = Math.Max(0, prefs.MaxSelect - selectedCards.Count);
        state["requires_manual_confirmation"] = prefs.RequireManualConfirmation;
        bool selectionQuotaReached = prefs.MaxSelect > 0 && selectedCards.Count >= prefs.MaxSelect;
        bool selectionWithinQuota = selectedCards.Count >= prefs.MinSelect && selectedCards.Count <= prefs.MaxSelect;

        // Preview container showing? (selection complete, awaiting confirm)
        // Upgrade screens use UpgradeSinglePreviewContainer / UpgradeMultiPreviewContainer
        // Enchant screens use EnchantSinglePreviewContainer / EnchantMultiPreviewContainer.
        var previewSingle = screen.GetNodeOrNull<Godot.Control>("%UpgradeSinglePreviewContainer");
        var previewMulti = screen.GetNodeOrNull<Godot.Control>("%UpgradeMultiPreviewContainer");
        var enchantPreviewSingle = screen.GetNodeOrNull<Godot.Control>("%EnchantSinglePreviewContainer");
        var enchantPreviewMulti = screen.GetNodeOrNull<Godot.Control>("%EnchantMultiPreviewContainer");
        var previewGeneric = screen.GetNodeOrNull<Godot.Control>("%PreviewContainer");
        bool previewShowing = (previewSingle?.Visible ?? false)
                            || (previewMulti?.Visible ?? false)
                            || (enchantPreviewSingle?.Visible ?? false)
                            || (enchantPreviewMulti?.Visible ?? false)
                            || (previewGeneric?.Visible ?? false);
        // Button states
        var closeButton = screen.GetNodeOrNull<NBackButton>("%Close");
        bool canCancel = closeButton?.IsEnabled ?? false;
        foreach (var container in new[] { previewSingle, previewMulti, enchantPreviewSingle, enchantPreviewMulti, previewGeneric })
        {
            if (container?.Visible != true)
                continue;

            var cancel = container.GetNodeOrNull<NBackButton>("Cancel")
                         ?? container.GetNodeOrNull<NBackButton>("%PreviewCancel");
            if (cancel?.IsEnabled == true)
            {
                canCancel = true;
                break;
            }
        }

        // Confirm button — search all preview containers and main screen
        bool canConfirm = false;
        foreach (var container in new[] { previewSingle, previewMulti, enchantPreviewSingle, enchantPreviewMulti, previewGeneric })
        {
            if (container?.Visible == true)
            {
                var confirm = container.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? container.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
                if (confirm?.IsEnabled == true) { canConfirm = true; break; }
            }
        }
        if (!canConfirm)
        {
            var mainConfirm = screen.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? screen.GetNodeOrNull<NConfirmButton>("%Confirm");
            if (mainConfirm?.IsEnabled == true) canConfirm = true;
        }
        // Fallback: search entire screen tree for any enabled confirm button
        // (covers subclasses like NDeckEnchantSelectScreen)
        if (!canConfirm)
        {
            canConfirm = FindAll<NConfirmButton>(screen).Any(b => b.IsEnabled && b.IsVisibleInTree());
        }
        canConfirm = canConfirm && selectionWithinQuota;
        if (!previewShowing && selectionQuotaReached && canConfirm
            && screen is NDeckEnchantSelectScreen or NDeckCardSelectScreen or NDeckUpgradeSelectScreen)
        {
            previewShowing = true;
        }
        if (!canCancel && selectionQuotaReached && selectedCards.Count > 0
            && screen is NDeckEnchantSelectScreen or NDeckCardSelectScreen or NDeckUpgradeSelectScreen)
        {
            canCancel = true;
        }
        state["preview_showing"] = previewShowing;
        state["can_cancel"] = canCancel;
        state["can_confirm"] = canConfirm;

        return state;
    }

    private static Dictionary<string, object?> BuildChooseCardState(NChooseACardSelectionScreen screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();
        state["screen_type"] = "choose";

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        state["prompt"] = "Choose a card.";

        var cardHolders = FindAllSortedByPosition<NGridCardHolder>(screen);
        var cards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in cardHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            cards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["description"] = SafeGetCardDescription(card, PileType.None),
                ["rarity"] = card.Rarity.ToString(),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = cards;

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        state["can_skip"] = skipButton?.IsEnabled == true && skipButton.Visible;
        state["preview_showing"] = false;
        state["can_confirm"] = false;
        state["can_cancel"] = state["can_skip"];

        return state;
    }

    private static Dictionary<string, object?> BuildHandSelectState(NPlayerHand hand, RunState runState)
    {
        var state = new Dictionary<string, object?>();
        var selectedCards = GetSelectedCards(hand);
        TryGetCardSelectorPrefs(hand, out var prefs);

        // Mode
        state["mode"] = hand.CurrentMode switch
        {
            NPlayerHand.Mode.SimpleSelect => "simple_select",
            NPlayerHand.Mode.UpgradeSelect => "upgrade_select",
            _ => hand.CurrentMode.ToString()
        };

        // Prompt text from %SelectionHeader
        var headerLabel = hand.GetNodeOrNull<Godot.Control>("%SelectionHeader");
        if (headerLabel != null)
        {
            var textVariant = headerLabel.Get("text");
            string? prompt = textVariant.VariantType != Godot.Variant.Type.Nil
                ? StripRichTextTags(textVariant.AsString())
                : null;
            state["prompt"] = prompt;
        }

        // Selectable cards (visible holders in the hand)
        var selectableCards = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in hand.ActiveHolders)
        {
            var card = holder.CardModel;
            if (card == null) continue;

            selectableCards.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["description"] = SafeGetCardDescription(card),
                ["is_upgraded"] = card.IsUpgraded,
                ["keywords"] = BuildHoverTips(card.HoverTips)
            });
            index++;
        }
        state["cards"] = selectableCards;

        state["selected_cards"] = BuildCompactCardList(selectedCards);
        state["selected_count"] = selectedCards.Count;
        state["min_select"] = prefs.MinSelect;
        state["max_select"] = prefs.MaxSelect;
        state["remaining_picks"] = Math.Max(0, prefs.MaxSelect - selectedCards.Count);
        state["requires_manual_confirmation"] = prefs.RequireManualConfirmation;

        // Confirm button state
        var confirmBtn = hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton");
        state["can_confirm"] = confirmBtn?.IsEnabled ?? false;

        return state;
    }

    private static bool TryGetPrivateField(object? target, string fieldName, out object? value)
    {
        value = null;
        if (target == null)
            return false;

        for (Type? type = target.GetType(); type != null; type = type.BaseType)
        {
            var field = type.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (field == null)
                continue;

            value = field.GetValue(target);
            return true;
        }

        return false;
    }

    private static bool TryGetCardSelectorPrefs(object? target, out CardSelectorPrefs prefs)
    {
        prefs = default;
        if (!TryGetPrivateField(target, "_prefs", out var raw) || raw is not CardSelectorPrefs typed)
            return false;

        prefs = typed;
        return true;
    }

    private static List<CardModel> GetSelectedCards(object? target)
    {
        if (TryGetPrivateField(target, "_selectedCards", out var raw) && raw is IEnumerable<CardModel> cards)
            return cards.ToList();

        return new List<CardModel>();
    }

    private static List<Dictionary<string, object?>> BuildCompactCardList(IEnumerable<CardModel> cards)
    {
        var list = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var card in cards)
        {
            list.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = card.Id.Entry,
                ["name"] = SafeGetText(() => card.Title),
                ["type"] = card.Type.ToString(),
                ["cost"] = card.EnergyCost.CostsX ? "X" : card.EnergyCost.GetAmountToSpend().ToString(),
                ["is_upgraded"] = card.IsUpgraded
            });
            index++;
        }
        return list;
    }

    private static Dictionary<string, object?> BuildRelicSelectState(NChooseARelicSelection screen, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        state["prompt"] = "Choose a relic.";

        var relicHolders = FindAll<NRelicBasicHolder>(screen);
        var relics = new List<Dictionary<string, object?>>();
        int index = 0;
        foreach (var holder in relicHolders)
        {
            var relic = holder.Relic?.Model;
            if (relic == null) continue;

            relics.Add(new Dictionary<string, object?>
            {
                ["index"] = index,
                ["id"] = relic.Id.Entry,
                ["name"] = SafeGetText(() => relic.Title),
                ["description"] = SafeGetText(() => relic.DynamicDescription),
                ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
            });
            index++;
        }
        state["relics"] = relics;

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        state["can_skip"] = skipButton?.IsEnabled == true && skipButton.Visible;

        return state;
    }

    private static Dictionary<string, object?> BuildTreasureState(TreasureRoom treasureRoom, RunState runState)
    {
        var state = new Dictionary<string, object?>();

        var player = LocalContext.GetMe(runState);
        if (player != null)
            state["player"] = BuildNonCombatPlayerState(player);
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);

        if (treasureUI == null)
        {
            state["message"] = "Treasure room loading...";
            return state;
        }

        // Auto-open chest if not yet opened
        var chestButton = treasureUI.GetNodeOrNull<NClickableControl>("Chest");
        if (chestButton is { IsEnabled: true })
        {
            chestButton.ForceClick();
            state["message"] = "Opening chest...";
            return state;
        }

        // Show relics available for picking
        var relicCollection = treasureUI.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
        if (relicCollection?.Visible == true)
        {
            var holders = FindAll<NTreasureRoomRelicHolder>(relicCollection)
                .Where(h => h.IsEnabled && h.Visible)
                .ToList();

            var relics = new List<Dictionary<string, object?>>();
            int index = 0;
            foreach (var holder in holders)
            {
                var relic = holder.Relic?.Model;
                if (relic == null) continue;
                relics.Add(new Dictionary<string, object?>
                {
                    ["index"] = index,
                    ["id"] = relic.Id.Entry,
                    ["name"] = SafeGetText(() => relic.Title),
                    ["description"] = SafeGetText(() => relic.DynamicDescription),
                    ["rarity"] = relic.Rarity.ToString(),
                    ["keywords"] = BuildHoverTips(relic.HoverTipsExcludingRelic)
                });
                index++;
            }
            state["relics"] = relics;
        }

        state["can_proceed"] = treasureUI.ProceedButton?.IsEnabled ?? false;

        return state;
    }

    private static string GetRewardTypeName(Reward reward) => reward switch
    {
        GoldReward => "gold",
        PotionReward => "potion",
        RelicReward => "relic",
        CardReward => "card",
        SpecialCardReward => "special_card",
        CardRemovalReward => "card_removal",
        _ => reward.GetType().Name.ToLower()
    };

    private static string BuildRewardKey(Reward reward)
    {
        string rewardType = GetRewardTypeName(reward);
        string label = SafeGetText(() => reward.Description);
        string specificId = reward switch
        {
            PotionReward potionReward => potionReward.Potion?.Id.Entry ?? string.Empty,
            CardReward => "card_reward",
            CardRemovalReward => "card_remove",
            GoldReward goldReward => goldReward.Amount.ToString(),
            RelicReward => "relic_reward",
            SpecialCardReward => "special_card_reward",
            _ => string.Empty
        };
        return $"{rewardType}|{specificId}|{label}".ToLowerInvariant();
    }

    private static bool IsRewardClaimable(Reward reward, Player? player, out string? claimBlockReason)
    {
        claimBlockReason = null;
        if (reward is PotionReward && player != null && !player.HasOpenPotionSlots)
        {
            claimBlockReason = "potion_slots_full";
            return false;
        }
        return true;
    }

    private static string GetRewardSourceName(RunState runState)
    {
        return runState.CurrentRoom switch
        {
            CombatRoom combatRoom when combatRoom.ParentEventId != null => "event_combat_end",
            CombatRoom => "combat_end",
            EventRoom => "event",
            TreasureRoom => "treasure",
            MerchantRoom => "shop",
            RestSiteRoom => "rest_site",
            null => "unknown",
            _ => runState.CurrentRoom.RoomType.ToString().ToLowerInvariant()
        };
    }

    private static List<Dictionary<string, object?>> BuildPowersState(Creature creature)
    {
        var powers = new List<Dictionary<string, object?>>();
        foreach (var power in creature.Powers)
        {
            if (!power.IsVisible) continue;

            // HoverTips resolves all dynamic vars (Amount, DynamicVars, etc.)
            // The first tip is the power's own description; the rest are extra keywords
            var allTips = power.HoverTips.ToList();
            string? resolvedDesc = null;
            var extraTips = new List<IHoverTip>();
            foreach (var tip in allTips)
            {
                if (tip.Id == power.Id.ToString())
                {
                    // This is the power's own hover tip — extract its resolved description
                    if (tip is HoverTip ht)
                        resolvedDesc = StripRichTextTags(ht.Description);
                }
                else
                {
                    extraTips.Add(tip);
                }
            }
            // Fallback to raw SmartDescription if HoverTips extraction failed
            resolvedDesc ??= SafeGetText(() => power.SmartDescription);

            powers.Add(new Dictionary<string, object?>
            {
                ["id"] = power.Id.Entry,
                ["name"] = SafeGetText(() => power.Title),
                ["amount"] = power.DisplayAmount,
                ["type"] = power.Type.ToString(),
                ["description"] = resolvedDesc,
                ["keywords"] = BuildHoverTips(extraTips)
            });
        }
        return powers;
    }
}
