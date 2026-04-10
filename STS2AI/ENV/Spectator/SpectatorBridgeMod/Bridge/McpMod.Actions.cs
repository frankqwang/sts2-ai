using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Rewards;
using MegaCrit.Sts2.Core.Nodes.Screens;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Screens.Shops;
using MegaCrit.Sts2.Core.Nodes.Screens.TreasureRoomRelic;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Nodes.Events;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Potions;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;

namespace STS2_MCP;

public static partial class McpMod
{
    private static Dictionary<string, object?> ExecuteAction(string action, Dictionary<string, JsonElement> data)
    {
        if (!RunManager.Instance.IsInProgress)
        {
            return action switch
            {
                "select_character" => ExecuteSelectCharacter(data),
                "set_ascension" => ExecuteSetAscension(data),
                "start_run" => ExecuteStartRun(data),
                _ => Error("No run in progress")
            };
        }

        if (action is "select_character" or "set_ascension" or "start_run")
            return Error("Menu action unavailable while a run is in progress");

        var runState = RunManager.Instance.DebugOnlyGetState()!;
        var player = LocalContext.GetMe(runState);
        if (player == null)
            return Error("Could not find local player");

        return action switch
        {
            "play_card" => ExecutePlayCard(player, data),
            "use_potion" => ExecuteUsePotion(player, data),
            "end_turn" => ExecuteEndTurn(player),
            "choose_map_node" => ExecuteChooseMapNode(data),
            "choose_event_option" => ExecuteChooseEventOption(data),
            "advance_dialogue" => ExecuteAdvanceDialogue(),
            "choose_rest_option" => ExecuteChooseRestOption(data),
            "shop_purchase" => ExecuteShopPurchase(player, data),
            "claim_reward" => ExecuteClaimReward(data),
            "select_card_reward" => ExecuteSelectCardReward(data),
            "skip_card_reward" => ExecuteSkipCardReward(),
            "proceed" => ExecuteProceedCompat(),
            "select_card" => ExecuteSelectCard(data),
            "confirm_selection" => ExecuteConfirmSelection(),
            "cancel_selection" => ExecuteCancelSelection(),
            "combat_select_card" => ExecuteCombatSelectCard(data),
            "combat_confirm_selection" => ExecuteCombatConfirmSelection(),
            "select_relic" => ExecuteSelectRelic(data),
            "skip_relic_selection" => ExecuteSkipRelicSelection(),
            "claim_treasure_relic" => ExecuteClaimTreasureRelic(data),
            "overlay_press" => ExecuteOverlayPress(data),
            _ => Error($"Unknown action: {action}")
        };
    }

    private static Dictionary<string, object?> ExecuteSelectCharacter(Dictionary<string, JsonElement> data)
    {
        var mainMenu = NGame.Instance?.MainMenu;
        if (mainMenu == null || !mainMenu.IsVisibleInTree())
            return Error("Main menu is not active");

        if (!TryEnsureCharacterSelectOpen(mainMenu, out var charSelectScreen, out var error))
            return Error(error);
        var screen = charSelectScreen!;

        var buttonContainer = screen.GetNodeOrNull<Godot.Node>("CharSelectButtons/ButtonContainer");
        if (buttonContainer == null)
            return Error("Character select button container not found");

        var buttons = FindAll<NCharacterSelectButton>(buttonContainer);
        if (buttons.Count == 0)
            return Error("No character buttons available");

        foreach (var button in buttons)
            button.UnlockIfPossible();

        NCharacterSelectButton? target = null;
        if (data.TryGetValue("character_id", out var characterIdElem))
        {
            string? characterId = characterIdElem.GetString()?.Trim();
            if (!string.IsNullOrWhiteSpace(characterId))
            {
                target = buttons.FirstOrDefault(b =>
                    string.Equals(b.Character.Id.Entry, characterId, StringComparison.OrdinalIgnoreCase)
                    || string.Equals(SafeGetText(() => b.Character.Title), characterId, StringComparison.OrdinalIgnoreCase));
            }
        }
        else if (data.TryGetValue("index", out var indexElem))
        {
            int index = indexElem.GetInt32();
            if (index < 0 || index >= buttons.Count)
                return Error($"Character index {index} out of range ({buttons.Count} buttons)");
            target = buttons[index];
        }
        else
        {
            return Error("Missing 'character_id' or 'index'");
        }

        if (target == null)
            return Error("Requested character not found");
        if (target.IsLocked)
            return Error($"Character '{target.Character.Id.Entry}' is locked");

        target.Select();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selected character: {target.Character.Id.Entry}",
            ["character_id"] = target.Character.Id.Entry,
            ["ascension"] = screen.Lobby.Ascension
        };
    }

    private static Dictionary<string, object?> ExecuteSetAscension(Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("ascension", out var ascensionElem))
            return Error("Missing 'ascension'");

        int requestedAscension = ascensionElem.GetInt32();
        if (requestedAscension < 0)
            return Error("Ascension must be >= 0");

        var mainMenu = NGame.Instance?.MainMenu;
        if (mainMenu == null || !mainMenu.IsVisibleInTree())
            return Error("Main menu is not active");

        if (!TryEnsureCharacterSelectOpen(mainMenu, out var charSelectScreen, out var error))
            return Error(error);
        var screen = charSelectScreen!;

        var ascensionPanel = screen.GetNodeOrNull<NAscensionPanel>("%AscensionPanel");
        if (ascensionPanel == null)
            return Error("Ascension panel not found");

        int maxAscension = Math.Max(0, screen.Lobby.MaxAscension);
        int appliedAscension = Math.Clamp(requestedAscension, 0, maxAscension);
        ascensionPanel.SetAscensionLevel(appliedAscension);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Ascension set to {appliedAscension}",
            ["requested_ascension"] = requestedAscension,
            ["ascension"] = appliedAscension,
            ["max_ascension"] = maxAscension
        };
    }

    private static Dictionary<string, object?> ExecuteStartRun(Dictionary<string, JsonElement> data)
    {
        if (RunManager.Instance.IsInProgress)
            return Error("Run already in progress");

        var game = NGame.Instance;
        var mainMenu = game?.MainMenu;
        if (game == null || mainMenu == null || !mainMenu.IsVisibleInTree())
            return Error("Main menu is not active");

        var unlockState = SaveManager.Instance.GenerateUnlockStateFromProgress();
        var charSelectScreen = TryGetCharacterSelectScreen(mainMenu);

        CharacterModel? character = null;
        if (data.TryGetValue("character_id", out var characterIdElem))
        {
            string? requestedId = characterIdElem.GetString();
            if (string.IsNullOrWhiteSpace(requestedId))
                return Error("'character_id' is empty");
            character = ResolveCharacter(requestedId);
            if (character == null)
                return Error($"Unknown character_id '{requestedId}'");
        }
        else if (charSelectScreen?.Visible == true)
        {
            character = charSelectScreen.Lobby.LocalPlayer.character;
        }

        character ??= ModelDb.Character<Ironclad>();

        if (!unlockState.Characters.Contains(character))
            return Error($"Character '{character.Id.Entry}' is locked");

        int requestedAscension = charSelectScreen?.Visible == true ? charSelectScreen.Lobby.Ascension : 0;
        if (data.TryGetValue("ascension", out var ascensionElem))
            requestedAscension = ascensionElem.GetInt32();

        if (requestedAscension < 0)
            return Error("Ascension must be >= 0");

        int maxAscension = Math.Max(0, SaveManager.Instance.Progress.GetOrCreateCharacterStats(character.Id).MaxAscension);
        int ascension = Math.Clamp(requestedAscension, 0, maxAscension);

        string seed;
        if (data.TryGetValue("seed", out var seedElem))
        {
            string? requestedSeed = seedElem.GetString();
            if (string.IsNullOrWhiteSpace(requestedSeed))
                return Error("'seed' is empty");
            seed = SeedHelper.CanonicalizeSeed(requestedSeed);
        }
        else
        {
            seed = game.DebugSeedOverride ?? SeedHelper.GetRandomSeed();
        }

        var acts = ActModel.GetRandomList(seed, unlockState, isMultiplayer: false).ToList();
        TaskHelper.RunSafely(game.StartNewSingleplayerRun(character, shouldSave: true, acts, new List<ModifierModel>(), seed, ascension));

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Starting run as {character.Id.Entry} at ascension {ascension}",
            ["character_id"] = character.Id.Entry,
            ["ascension"] = ascension,
            ["seed"] = seed
        };
    }

    private static bool TryEnsureCharacterSelectOpen(
        MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenu mainMenu,
        out NCharacterSelectScreen? charSelectScreen,
        out string error)
    {
        charSelectScreen = TryGetCharacterSelectScreen(mainMenu);
        if (charSelectScreen?.Visible == true)
        {
            error = string.Empty;
            return true;
        }

        var singleplayerButton = mainMenu.GetNodeOrNull<NButton>("MainMenuTextButtons/SingleplayerButton");
        if (singleplayerButton is { Visible: true, IsEnabled: true })
            singleplayerButton.ForceClick();

        charSelectScreen = TryGetCharacterSelectScreen(mainMenu);
        if (charSelectScreen?.Visible == true)
        {
            error = string.Empty;
            return true;
        }

        var standardButton = mainMenu.GetNodeOrNull<NButton>("Submenus/SingleplayerSubmenu/StandardButton");
        if (standardButton is { Visible: true, IsEnabled: true })
            standardButton.ForceClick();

        charSelectScreen = TryGetCharacterSelectScreen(mainMenu);
        if (charSelectScreen?.Visible == true)
        {
            error = string.Empty;
            return true;
        }

        error = "Unable to open character select screen from the main menu";
        return false;
    }

    private static NCharacterSelectScreen? TryGetCharacterSelectScreen(MegaCrit.Sts2.Core.Nodes.Screens.MainMenu.NMainMenu mainMenu)
    {
        return mainMenu.GetNodeOrNull<NCharacterSelectScreen>("Submenus/CharacterSelectScreen");
    }

    private static CharacterModel? ResolveCharacter(string characterIdOrName)
    {
        string value = characterIdOrName.Trim();
        if (value.Length == 0)
            return null;

        var match = ModelDb.AllCharacters.FirstOrDefault(c =>
            string.Equals(c.Id.Entry, value, StringComparison.OrdinalIgnoreCase)
            || string.Equals(SafeGetText(() => c.Title), value, StringComparison.OrdinalIgnoreCase));
        if (match != null)
            return match;

        var randomCharacter = ModelDb.Character<RandomCharacter>();
        if (string.Equals(randomCharacter.Id.Entry, value, StringComparison.OrdinalIgnoreCase)
            || string.Equals(SafeGetText(() => randomCharacter.Title), value, StringComparison.OrdinalIgnoreCase))
            return randomCharacter;

        return null;
    }

    private static Dictionary<string, object?> ExecutePlayCard(Player player, Dictionary<string, JsonElement> data)
    {
        if (!CombatManager.Instance.IsInProgress)
            return Error("Not in combat");
        if (!CombatManager.Instance.IsPlayPhase)
            return Error("Not in play phase — cannot act during enemy turn");
        if (CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled");
        if (!player.Creature.IsAlive)
            return Error("Player creature is dead — cannot play cards");

        var combatState = player.Creature.CombatState;
        if (combatState == null)
            return Error("No combat state");

        // Get card by index in hand
        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index'");

        int cardIndex = indexElem.GetInt32();
        var hand = player.PlayerCombatState?.Hand;
        if (hand == null)
            return Error("No hand available");

        if (cardIndex < 0 || cardIndex >= hand.Cards.Count)
            return Error($"card_index {cardIndex} out of range (hand has {hand.Cards.Count} cards)");

        var card = hand.Cards[cardIndex];

        if (!card.CanPlay(out var reason, out _))
            return Error($"Card '{card.Title}' cannot be played: {reason}");

        // Resolve target
        Creature? target = null;
        if (card.TargetType == TargetType.AnyEnemy)
        {
            if (!TryGetTargetIdentifier(data, out string targetId))
                return Error("Card requires a target. Provide 'target' or 'target_id'.");

            target = ResolveTarget(combatState, targetId);
            if (target == null)
                return Error($"Target '{targetId}' not found among alive enemies");
        }

        // Play the card via the action queue (same path as the game UI)
        RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(new PlayCardAction(card, target));

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Playing '{card.Title}'" + (target != null ? $" targeting {SafeGetText(() => target.Monster?.Title) ?? "target"}" : "")
        };
    }

    private static Dictionary<string, object?> ExecuteEndTurn(Player player)
    {
        if (!CombatManager.Instance.IsInProgress)
            return Error("Not in combat");
        if (!CombatManager.Instance.IsPlayPhase)
            return Error("Not in play phase — cannot act during enemy turn");
        if (CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled (turn may already be ending)");

        // Match the game's own CanTurnBeEnded guard (NEndTurnButton.cs:114-123)
        var hand = NCombatRoom.Instance?.Ui?.Hand;
        if (hand != null && (hand.InCardPlay || hand.CurrentMode != NPlayerHand.Mode.Play))
            return Error("Cannot end turn while a card is being played or hand is in selection mode");

        PlayerCmd.EndTurn(player, canBackOut: false);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Ending turn"
        };
    }

    private static Dictionary<string, object?> ExecuteUsePotion(Player player, Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("slot", out var slotElem))
            return Error("Missing 'slot' (potion slot index)");

        int slot = slotElem.GetInt32();
        if (slot < 0 || slot >= player.PotionSlots.Count)
            return Error($"Potion slot {slot} out of range (player has {player.PotionSlots.Count} slots)");

        var potion = player.GetPotionAtSlotIndex(slot);
        if (potion == null)
            return Error($"No potion in slot {slot}");
        if (potion.IsQueued)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' is already queued for use");
        if (potion.Owner.Creature.IsDead)
            return Error("Cannot use potion — player creature is dead");
        if (!potion.PassesCustomUsabilityCheck)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' cannot be used right now");

        bool inCombat = CombatManager.Instance.IsInProgress;
        if (potion.Usage == PotionUsage.CombatOnly)
        {
            if (!inCombat)
                return Error($"Potion '{SafeGetText(() => potion.Title)}' can only be used in combat");
            if (!CombatManager.Instance.IsPlayPhase)
                return Error("Cannot use potions outside of play phase");
        }
        else if (potion.Usage == PotionUsage.Automatic)
            return Error($"Potion '{SafeGetText(() => potion.Title)}' is automatic and cannot be manually used");

        if (inCombat && CombatManager.Instance.PlayerActionsDisabled)
            return Error("Player actions are currently disabled");

        // Resolve target
        Creature? target = null;
        var combatState = player.Creature.CombatState;

        switch (potion.TargetType)
        {
            case TargetType.AnyEnemy:
                if (!TryGetTargetIdentifier(data, out string targetId))
                    return Error("Potion requires a target enemy. Provide 'target' or 'target_id'.");
                if (combatState == null)
                    return Error("No combat state for target resolution");
                target = ResolveTarget(combatState, targetId);
                if (target == null)
                    return Error($"Target '{targetId}' not found among alive enemies");
                break;
            case TargetType.Self:
            case TargetType.AnyAlly:
            case TargetType.AnyPlayer:
                target = player.Creature;
                break;
            default:
                target = null;
                break;
        }

        potion.EnqueueManualUse(target);

        string targetMsg = potion.TargetType switch
        {
            TargetType.AnyEnemy => $" targeting {SafeGetText(() => target?.Monster?.Title) ?? "enemy"}",
            TargetType.Self or TargetType.AnyPlayer or TargetType.AnyAlly => " on self",
            _ => ""
        };

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Using potion '{SafeGetText(() => potion.Title)}' from slot {slot}{targetMsg}"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseEventOption(Dictionary<string, JsonElement> data)
    {
        var uiRoom = NEventRoom.Instance;
        if (uiRoom == null)
            return Error("Event room is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (event option index)");

        int index = indexElem.GetInt32();

        var buttons = FindAll<NEventOptionButton>(uiRoom)
            .Where(b => !b.Option.IsLocked)
            .ToList();

        if (buttons.Count == 0)
            return Error("No unlocked event options available");
        if (index < 0 || index >= buttons.Count)
            return Error($"Event option index {index} out of range ({buttons.Count} unlocked options)");

        var button = buttons[index];
        string title = SafeGetText(() => button.Option.Title) ?? "option";
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Choosing event option: {title}"
        };
    }

    private static Dictionary<string, object?> ExecuteAdvanceDialogue()
    {
        var uiRoom = NEventRoom.Instance;
        if (uiRoom == null)
            return Error("Event room is not open");

        var ancientLayout = FindFirst<NAncientEventLayout>(uiRoom);
        if (ancientLayout == null)
            return Error("No ancient dialogue active");

        var hitbox = ancientLayout.GetNodeOrNull<NClickableControl>("%DialogueHitbox");
        if (hitbox == null || !hitbox.Visible || !hitbox.IsEnabled)
            return Error("Dialogue hitbox not available — dialogue may have ended");

        hitbox.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Advancing dialogue"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseRestOption(Dictionary<string, JsonElement> data)
    {
        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (rest site option index)");

        int index = indexElem.GetInt32();

        var restRoom = NRestSiteRoom.Instance
            ?? FindFirst<NRestSiteRoom>(((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (restRoom == null)
            return Error("Rest site room is not open");

        var buttons = FindAll<NRestSiteButton>(restRoom)
            .Where(b => b.Option.IsEnabled)
            .ToList();

        if (index < 0 || index >= buttons.Count)
            return Error($"Rest option index {index} out of range ({buttons.Count} enabled options)");

        var button = buttons[index];
        string optionName = SafeGetText(() => button.Option.Title) ?? button.Option.OptionId;
        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting rest site option: {optionName}"
        };
    }

    private static Dictionary<string, object?> ExecuteShopPurchase(Player player, Dictionary<string, JsonElement> data)
    {
        if (player.RunState.CurrentRoom is not MerchantRoom merchantRoom)
            return Error("Not in a shop");

        // Auto-open inventory if needed
        var merchUI = NMerchantRoom.Instance;
        if (merchUI != null && !merchUI.Inventory.IsOpen)
            merchUI.OpenInventory();

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (shop item index)");

        int index = indexElem.GetInt32();

        var allEntries = merchantRoom.Inventory.AllEntries.ToList();
        if (index < 0 || index >= allEntries.Count)
            return Error($"Shop item index {index} out of range ({allEntries.Count} items)");

        var entry = allEntries[index];
        if (!entry.IsStocked)
            return Error("Item is sold out");
        if (!entry.EnoughGold)
            return Error($"Not enough gold (need {entry.Cost}, have {player.Gold})");

        // Fire-and-forget purchase (same path as AutoSlay)
        _ = entry.OnTryPurchaseWrapper(merchantRoom.Inventory);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Purchasing item for {entry.Cost} gold"
        };
    }

    private static Dictionary<string, object?> ExecuteChooseMapNode(Dictionary<string, JsonElement> data)
    {
        var mapScreen = NMapScreen.Instance;
        if (mapScreen == null || !mapScreen.IsOpen)
            return Error("Map screen is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (map node index from next_options)");

        int index = indexElem.GetInt32();

        var travelable = FindAll<NMapPoint>(mapScreen)
            .Where(mp => mp.State == MapPointState.Travelable)
            .OrderBy(mp => mp.Point.coord.col)
            .ToList();

        if (travelable.Count == 0)
            return Error("No travelable map nodes available");
        if (index < 0 || index >= travelable.Count)
            return Error($"Map node index {index} out of range ({travelable.Count} options available)");

        var target = travelable[index];
        mapScreen.OnMapPointSelectedLocally(target);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Traveling to {target.Point.PointType} at ({target.Point.coord.col},{target.Point.coord.row})"
        };
    }

    private static Dictionary<string, object?> ExecuteClaimReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NRewardsScreen rewardsScreen)
            return Error("Rewards screen is not open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (reward index)");

        int index = indexElem.GetInt32();

        var enabledButtons = FindAll<NRewardButton>(rewardsScreen)
            .Where(b => b.IsEnabled && b.Reward != null)
            .ToList();

        if (index < 0 || index >= enabledButtons.Count)
            return Error($"Reward index {index} out of range (screen has {enabledButtons.Count} claimable rewards)");

        var button = enabledButtons[index];
        var reward = button.Reward!;
        string rewardDesc = GetRewardTypeName(reward);
        if (reward is GoldReward g)
            rewardDesc = $"gold ({g.Amount})";
        else if (reward is PotionReward p)
            rewardDesc = $"potion ({SafeGetText(() => p.Potion?.Title)})";
        else if (reward is CardReward)
            rewardDesc = "card (opens card selection)";

        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Claiming reward: {rewardDesc}"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectCardReward(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCardRewardSelectionScreen cardScreen)
            return Error("Card reward selection screen is not open");

        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index'");

        int cardIndex = indexElem.GetInt32();

        var cardHolders = FindAllSortedByPosition<NCardHolder>(cardScreen);
        if (cardIndex < 0 || cardIndex >= cardHolders.Count)
            return Error($"Card index {cardIndex} out of range (screen has {cardHolders.Count} cards)");

        var holder = cardHolders[cardIndex];
        string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting card: {cardName}"
        };
    }

    private static Dictionary<string, object?> ExecuteSkipCardReward()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NCardRewardSelectionScreen cardScreen)
            return Error("Card reward selection screen is not open");

        var altButtons = FindAll<NCardRewardAlternativeButton>(cardScreen);
        if (altButtons.Count == 0)
            return Error("No skip option available on this card reward");

        altButtons[0].ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Skipping card reward"
        };
    }

    private static Dictionary<string, object?> ExecuteProceed()
    {
        // Try rewards overlay
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NRewardsScreen rewardsScreen)
        {
            var btn = FindFirst<NProceedButton>(rewardsScreen);
            if (btn is { IsEnabled: true })
            {
                btn.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from rewards" };
            }
        }

        // Try rest site
        if (NRestSiteRoom.Instance is { } restRoom && restRoom.ProceedButton.IsEnabled)
        {
            restRoom.ProceedButton.ForceClick();
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from rest site" };
        }

        // Try merchant — close inventory first if open, then proceed
        if (NMerchantRoom.Instance is { } merchRoom)
        {
            if (merchRoom.Inventory.IsOpen)
            {
                var closeMethod = merchRoom.Inventory.GetType().GetMethod("Close", BindingFlags.Instance | BindingFlags.NonPublic);
                if (closeMethod != null)
                {
                    closeMethod.Invoke(merchRoom.Inventory, null);
                    if (merchRoom.ProceedButton.IsEnabled)
                    {
                        merchRoom.ProceedButton.ForceClick();
                        return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Closed inventory and proceeded from shop" };
                    }
                    return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Closing shop inventory" };
                }

                var backBtn = FindFirst<NBackButton>(merchRoom);
                if (backBtn is { IsEnabled: true })
                {
                    backBtn.ForceClick();
                    return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Closing shop inventory" };
                }
            }
            if (merchRoom.ProceedButton.IsEnabled)
            {
                merchRoom.ProceedButton.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from shop" };
            }
        }

        // Try treasure room
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (treasureUI != null && treasureUI.ProceedButton.IsEnabled)
        {
            treasureUI.ProceedButton.ForceClick();
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Proceeding from treasure room" };
        }

        return Error("No proceed button available or enabled");
    }

    private static Dictionary<string, object?> ExecuteProceedCompat()
    {
        var currentScreen = ActiveScreenContext.Instance.GetCurrentScreen();
        if (currentScreen is NMerchantInventory activeInventory && TryCloseMerchantInventory(activeInventory))
            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Closing shop inventory" };

        if (NMerchantRoom.Instance is { } merchRoom && merchRoom.Inventory.IsOpen && TryCloseMerchantInventory(merchRoom.Inventory))
        {
            if (merchRoom.ProceedButton.IsEnabled)
            {
                merchRoom.ProceedButton.ForceClick();
                return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Closed inventory and proceeded from shop" };
            }

            return new Dictionary<string, object?> { ["status"] = "ok", ["message"] = "Closing shop inventory" };
        }

        return ExecuteProceed();
    }

    private static bool TryCloseMerchantInventory(NMerchantInventory inventory)
    {
        var closeMethod = inventory.GetType().GetMethod("Close", BindingFlags.Instance | BindingFlags.NonPublic);
        if (closeMethod != null)
        {
            closeMethod.Invoke(inventory, null);
            return true;
        }

        var backBtn = FindFirst<NBackButton>(inventory);
        if (backBtn is { IsEnabled: true })
        {
            backBtn.ForceClick();
            return true;
        }

        return false;
    }

    private static Dictionary<string, object?> ExecuteOverlayPress(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay == null)
            return Error("No overlay is open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (overlay button index)");

        int index = indexElem.GetInt32();

        var buttons = FindAll<NClickableControl>((Godot.Node)overlay)
            .Where(b => b.Visible && b.IsVisibleInTree())
            .ToList();

        if (index < 0 || index >= buttons.Count)
            return Error($"Overlay button index {index} out of range ({buttons.Count} visible buttons)");

        var button = buttons[index];
        if (!button.IsEnabled)
            return Error($"Overlay button {index} is disabled");

        button.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Pressing overlay button: {button.Name}"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectCard(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (card index in the grid)");

        int index = indexElem.GetInt32();

        if (overlay is NCardGridSelectionScreen gridScreen)
        {
            foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%EnchantSinglePreviewContainer", "%EnchantMultiPreviewContainer", "%PreviewContainer" })
            {
                var previewContainer = gridScreen.GetNodeOrNull<Godot.Control>(containerName);
                if (previewContainer?.Visible == true)
                    return Error("Selection preview is active — use confirm_selection or cancel_selection");
            }

            var grid = FindFirst<NCardGrid>(gridScreen);
            if (grid == null)
                return Error("Card grid not found in selection screen");

            var holders = FindAllSortedByPosition<NGridCardHolder>(grid);
            if (index < 0 || index >= holders.Count)
                return Error($"Card index {index} out of range ({holders.Count} cards available)");

            var holder = holders[index];
            string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
            if (holder.CardModel != null && TryInvokeCardGridSelection(gridScreen, holder.CardModel))
            {
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = $"Selecting card: {cardName}"
                };
            }
            grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder);
            holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Toggling card selection: {cardName}"
            };
        }
        else if (overlay is NChooseACardSelectionScreen chooseScreen)
        {
            var holders = FindAllSortedByPosition<NGridCardHolder>(chooseScreen);
            if (index < 0 || index >= holders.Count)
                return Error($"Card index {index} out of range ({holders.Count} cards available)");

            var holder = holders[index];
            string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";
            var grid = FindFirst<NCardGrid>(chooseScreen);
            if (grid != null)
                grid.EmitSignal(NCardGrid.SignalName.HolderPressed, holder);
            holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = $"Choosing card: {cardName}"
            };
        }

        return Error("No card selection screen is open");
    }

    private static bool TryInvokeCardGridSelection(NCardGridSelectionScreen screen, CardModel card)
    {
        var method = screen.GetType().GetMethod(
            "OnCardClicked",
            BindingFlags.Instance | BindingFlags.NonPublic,
            binder: null,
            types: new[] { typeof(CardModel) },
            modifiers: null);
        if (method == null)
            return false;

        method.Invoke(screen, new object[] { card });
        return true;
    }

    private static Dictionary<string, object?> ExecuteConfirmSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is NChooseACardSelectionScreen)
            return Error("Choose-a-card screen requires no confirmation — use select_card(index) to pick directly");
        if (overlay is not NCardGridSelectionScreen screen)
            return Error("No card selection screen is open");

        // Check all preview containers (upgrade uses UpgradeSinglePreviewContainer / UpgradeMultiPreviewContainer,
        // NDeckCardSelectScreen uses PreviewContainer with %PreviewConfirm)
        foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%EnchantSinglePreviewContainer", "%EnchantMultiPreviewContainer", "%PreviewContainer" })
        {
            var container = screen.GetNodeOrNull<Godot.Control>(containerName);
            if (container?.Visible == true)
            {
                var confirm = container.GetNodeOrNull<NConfirmButton>("Confirm")
                              ?? container.GetNodeOrNull<NConfirmButton>("%PreviewConfirm");
                if (confirm is { IsEnabled: true })
                {
                    confirm.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Confirming selection from preview"
                    };
                }
            }
        }

        // Try main confirm button
        var mainConfirm = screen.GetNodeOrNull<NConfirmButton>("Confirm")
                          ?? screen.GetNodeOrNull<NConfirmButton>("%Confirm");
        if (mainConfirm is { IsEnabled: true })
        {
            mainConfirm.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Confirming selection"
            };
        }

        // Fallback: find ANY enabled NConfirmButton in the screen tree.
        // Covers NCardGridSelectionScreen subclasses (like NDeckEnchantSelectScreen)
        // whose confirm button isn't in any of the known container paths above.
        var allConfirmButtons = FindAll<NConfirmButton>(screen);
        foreach (var btn in allConfirmButtons)
        {
            if (btn.IsEnabled && btn.IsVisibleInTree())
            {
                btn.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Confirming selection"
                };
            }
        }

        return Error("No confirm button is currently enabled — select more cards first");
    }

    private static Dictionary<string, object?> ExecuteCancelSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();

        // Handle choose-a-card screen (skip button)
        if (overlay is NChooseACardSelectionScreen chooseScreen)
        {
            var skipButton = chooseScreen.GetNodeOrNull<NClickableControl>("SkipButton");
            if (skipButton is { IsEnabled: true })
            {
                skipButton.ForceClick();
                return new Dictionary<string, object?>
                {
                    ["status"] = "ok",
                    ["message"] = "Skipping card choice"
                };
            }
            return Error("No skip option available — a card must be chosen");
        }

        if (overlay is not NCardGridSelectionScreen screen)
            return Error("No card selection screen is open");

        // If preview is showing, cancel back to selection
        foreach (var containerName in new[] { "%UpgradeSinglePreviewContainer", "%UpgradeMultiPreviewContainer", "%EnchantSinglePreviewContainer", "%EnchantMultiPreviewContainer", "%PreviewContainer" })
        {
            var container = screen.GetNodeOrNull<Godot.Control>(containerName);
            if (container?.Visible == true)
            {
                var cancelBtn = container.GetNodeOrNull<NBackButton>("Cancel")
                                ?? container.GetNodeOrNull<NBackButton>("%PreviewCancel");
                if (cancelBtn is { IsEnabled: true })
                {
                    cancelBtn.ForceClick();
                    return new Dictionary<string, object?>
                    {
                        ["status"] = "ok",
                        ["message"] = "Cancelling preview — returning to card selection"
                    };
                }
            }
        }

        // Close the screen entirely
        var closeButton = screen.GetNodeOrNull<NBackButton>("%Close");
        if (closeButton is { IsEnabled: true })
        {
            closeButton.ForceClick();
            return new Dictionary<string, object?>
            {
                ["status"] = "ok",
                ["message"] = "Closing card selection screen"
            };
        }

        return Error("No cancel/close button is currently enabled — selection may be mandatory");
    }

    private static Dictionary<string, object?> ExecuteCombatSelectCard(Dictionary<string, JsonElement> data)
    {
        var hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
            return Error("No in-combat card selection is active");

        if (!data.TryGetValue("card_index", out var indexElem))
            return Error("Missing 'card_index' (index of the card in hand)");

        int index = indexElem.GetInt32();
        var holders = hand.ActiveHolders;
        if (index < 0 || index >= holders.Count)
            return Error($"Card index {index} out of range ({holders.Count} selectable cards)");

        var holder = holders[index];
        string cardName = SafeGetText(() => holder.CardModel?.Title) ?? "unknown";

        // Emit the Pressed signal — same path the game UI uses
        holder.EmitSignal(NCardHolder.SignalName.Pressed, holder);

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting card from hand: {cardName}"
        };
    }

    private static Dictionary<string, object?> ExecuteCombatConfirmSelection()
    {
        var hand = NPlayerHand.Instance;
        if (hand == null || !hand.IsInCardSelection)
            return Error("No in-combat card selection is active");

        var confirmBtn = hand.GetNodeOrNull<NConfirmButton>("%SelectModeConfirmButton");
        if (confirmBtn == null || !confirmBtn.IsEnabled)
            return Error("Confirm button is not enabled — select more cards first");

        confirmBtn.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Confirming combat card selection"
        };
    }

    private static Dictionary<string, object?> ExecuteSelectRelic(Dictionary<string, JsonElement> data)
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseARelicSelection screen)
            return Error("No relic selection screen is open");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (relic index)");

        int index = indexElem.GetInt32();

        var holders = FindAll<NRelicBasicHolder>(screen);
        if (index < 0 || index >= holders.Count)
            return Error($"Relic index {index} out of range ({holders.Count} relics available)");

        var holder = holders[index];
        string relicName = SafeGetText(() => holder.Relic?.Model?.Title) ?? "unknown";
        holder.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Selecting relic: {relicName}"
        };
    }

    private static Dictionary<string, object?> ExecuteSkipRelicSelection()
    {
        var overlay = NOverlayStack.Instance?.Peek();
        if (overlay is not NChooseARelicSelection screen)
            return Error("No relic selection screen is open");

        var skipButton = screen.GetNodeOrNull<NClickableControl>("SkipButton");
        if (skipButton is not { IsEnabled: true })
            return Error("No skip option available");

        skipButton.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = "Skipping relic selection"
        };
    }

    private static Dictionary<string, object?> ExecuteClaimTreasureRelic(Dictionary<string, JsonElement> data)
    {
        var treasureUI = FindFirst<NTreasureRoom>(
            ((Godot.SceneTree)Godot.Engine.GetMainLoop()).Root);
        if (treasureUI == null)
            return Error("Treasure room is not open");

        var relicCollection = treasureUI.GetNodeOrNull<NTreasureRoomRelicCollection>("%RelicCollection");
        if (relicCollection?.Visible != true)
            return Error("Relic collection is not visible — chest may not be opened yet");

        if (!data.TryGetValue("index", out var indexElem))
            return Error("Missing 'index' (relic index)");

        int index = indexElem.GetInt32();

        var holders = FindAll<NTreasureRoomRelicHolder>(relicCollection)
            .Where(h => h.IsEnabled && h.Visible)
            .ToList();

        if (index < 0 || index >= holders.Count)
            return Error($"Relic index {index} out of range ({holders.Count} relics available)");

        var holder = holders[index];
        string relicName = SafeGetText(() => holder.Relic?.Model?.Title) ?? "unknown";
        holder.ForceClick();

        return new Dictionary<string, object?>
        {
            ["status"] = "ok",
            ["message"] = $"Claiming treasure relic: {relicName}"
        };
    }

    private static bool TryGetTargetIdentifier(Dictionary<string, JsonElement> data, out string targetId)
    {
        if (data.TryGetValue("target", out var targetElem) && targetElem.ValueKind == JsonValueKind.String)
        {
            targetId = targetElem.GetString() ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(targetId))
                return true;
        }

        if (data.TryGetValue("target_id", out var targetIdElem))
        {
            if (targetIdElem.ValueKind == JsonValueKind.Number)
            {
                targetId = targetIdElem.GetUInt32().ToString();
                return true;
            }

            if (targetIdElem.ValueKind == JsonValueKind.String)
            {
                targetId = targetIdElem.GetString() ?? string.Empty;
                if (!string.IsNullOrWhiteSpace(targetId))
                    return true;
            }
        }

        targetId = string.Empty;
        return false;
    }

    private static Creature? ResolveTarget(CombatState combatState, string entityId)
    {
        // Try to match by entity_id pattern: "model_entry_N"
        // First try matching by combat_id if it's a pure number
        if (uint.TryParse(entityId, out uint combatId))
            return combatState.GetCreature(combatId);

        // Match by entity_id pattern (e.g., "jaw_worm_0")
        // We rebuild the entity IDs the same way as BuildEnemyState
        var entityCounts = new Dictionary<string, int>();
        foreach (var creature in combatState.Enemies)
        {
            if (!creature.IsAlive) continue;
            string baseId = creature.Monster?.Id.Entry ?? "unknown";
            if (!entityCounts.TryGetValue(baseId, out int count))
                count = 0;
            entityCounts[baseId] = count + 1;
            string generatedId = $"{baseId}_{count}";

            if (generatedId == entityId)
                return creature;
        }

        return null;
    }
}
