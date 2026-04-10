using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text.Json;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Training;

namespace STS2_MCP;

public static partial class McpMod
{
    private static void HandleGetCombatEnvState(HttpListenerResponse response)
    {
        if (!IsCombatTrainerModeActive())
        {
            SendError(response, 409, "Combat trainer mode is not active. Launch the source-built game with --combat-trainer.");
            return;
        }

        try
        {
            var stateTask = RunOnMainThread(() => CombatTrainingEnvService.Instance.GetState());
            var state = stateTask.GetAwaiter().GetResult();
            SendJson(response, ShapeCombatEnvState(state));
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Failed to read combat env state: {ex.Message}");
        }
    }

    private static void HandlePostCombatEnvReset(HttpListenerRequest request, HttpListenerResponse response)
    {
        if (!IsCombatTrainerModeActive())
        {
            SendError(response, 409, "Combat trainer mode is not active. Launch the source-built game with --combat-trainer.");
            return;
        }

        try
        {
            CombatTrainingResetRequest resetRequest = ParseCombatEnvResetRequest(request);
            var stateTaskTask = RunOnMainThread(() => CombatTrainingEnvService.Instance.ResetAsync(resetRequest));
            var state = stateTaskTask.GetAwaiter().GetResult().GetAwaiter().GetResult();
            SendJson(response, ShapeCombatEnvState(state));
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}");
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Combat env reset failed: {ex.Message}");
        }
    }

    private static void HandlePostCombatEnvStep(HttpListenerRequest request, HttpListenerResponse response)
    {
        if (!IsCombatTrainerModeActive())
        {
            SendError(response, 409, "Combat trainer mode is not active. Launch the source-built game with --combat-trainer.");
            return;
        }

        try
        {
            CombatTrainingActionRequest actionRequest = ParseCombatEnvStepRequest(request);
            var resultTaskTask = RunOnMainThread(() => CombatTrainingEnvService.Instance.StepAsync(actionRequest));
            var result = resultTaskTask.GetAwaiter().GetResult().GetAwaiter().GetResult();
            SendJson(response, ShapeCombatEnvStepResult(result));
        }
        catch (JsonException ex)
        {
            SendError(response, 400, $"Invalid JSON: {ex.Message}");
        }
        catch (InvalidOperationException ex)
        {
            SendError(response, 400, ex.Message);
        }
        catch (Exception ex)
        {
            SendError(response, 500, $"Combat env step failed: {ex.Message}");
        }
    }

    private static CombatTrainingResetRequest ParseCombatEnvResetRequest(HttpListenerRequest request)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
        {
            body = reader.ReadToEnd();
        }

        if (string.IsNullOrWhiteSpace(body))
        {
            return new CombatTrainingResetRequest();
        }

        var parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body)
            ?? throw new JsonException("Request body must be a JSON object.");

        var resetRequest = new CombatTrainingResetRequest();
        if (parsed.TryGetValue("character_id", out var characterElem) && characterElem.ValueKind != JsonValueKind.Null)
        {
            resetRequest.CharacterId = characterElem.GetString();
        }
        if (parsed.TryGetValue("encounter_id", out var encounterElem) && encounterElem.ValueKind != JsonValueKind.Null)
        {
            resetRequest.EncounterId = encounterElem.GetString();
        }
        if (parsed.TryGetValue("seed", out var seedElem) && seedElem.ValueKind != JsonValueKind.Null)
        {
            resetRequest.Seed = seedElem.GetString();
        }
        if (parsed.TryGetValue("ascension_level", out var ascensionElem) && ascensionElem.ValueKind != JsonValueKind.Null)
        {
            resetRequest.AscensionLevel = ascensionElem.GetInt32();
        }
        return resetRequest;
    }

    private static CombatTrainingActionRequest ParseCombatEnvStepRequest(HttpListenerRequest request)
    {
        string body;
        using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
        {
            body = reader.ReadToEnd();
        }

        var parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(body)
            ?? throw new JsonException("Request body must be a JSON object.");

        if (!parsed.TryGetValue("type", out var typeElem) || typeElem.ValueKind != JsonValueKind.String)
        {
            throw new InvalidOperationException("Missing 'type'. Expected 'play_card', 'end_turn', 'select_hand_card', 'confirm_selection', or 'cancel_selection'.");
        }

        string type = typeElem.GetString() ?? "";
        var actionRequest = new CombatTrainingActionRequest
        {
            Type = type switch
            {
                "play_card" => CombatTrainingActionType.PlayCard,
                "end_turn" => CombatTrainingActionType.EndTurn,
                "select_hand_card" => CombatTrainingActionType.SelectHandCard,
                "confirm_selection" => CombatTrainingActionType.ConfirmSelection,
                "cancel_selection" => CombatTrainingActionType.CancelSelection,
                _ => throw new InvalidOperationException($"Unsupported combat env action type: '{type}'.")
            }
        };

        if (parsed.TryGetValue("hand_index", out var handIndexElem) && handIndexElem.ValueKind != JsonValueKind.Null)
        {
            actionRequest.HandIndex = handIndexElem.GetInt32();
        }

        if (parsed.TryGetValue("target_id", out var targetIdElem) && targetIdElem.ValueKind != JsonValueKind.Null)
        {
            actionRequest.TargetId = targetIdElem.ValueKind switch
            {
                JsonValueKind.Number => targetIdElem.GetUInt32(),
                JsonValueKind.String when uint.TryParse(targetIdElem.GetString(), out uint parsedTargetId) => parsedTargetId,
                _ => throw new InvalidOperationException("target_id must be a uint or a numeric string.")
            };
        }

        return actionRequest;
    }

    private static Dictionary<string, object?> ShapeCombatEnvStepResult(CombatTrainingStepResult result)
    {
        return new Dictionary<string, object?>
        {
            ["accepted"] = result.Accepted,
            ["error"] = result.Error,
            ["state"] = result.State != null ? ShapeCombatEnvState(result.State) : null
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvState(CombatTrainingStateSnapshot state)
    {
        return new Dictionary<string, object?>
        {
            ["is_trainer_active"] = state.IsTrainerActive,
            ["is_combat_active"] = state.IsCombatActive,
            ["is_episode_done"] = state.IsEpisodeDone,
            ["victory"] = state.Victory,
            ["episode_number"] = state.EpisodeNumber,
            ["seed"] = state.Seed,
            ["character_id"] = state.CharacterId,
            ["encounter_id"] = state.EncounterId,
            ["ascension_level"] = state.AscensionLevel,
            ["round_number"] = state.RoundNumber,
            ["current_side"] = state.CurrentSide.ToString().ToLowerInvariant(),
            ["is_play_phase"] = state.IsPlayPhase,
            ["player_actions_disabled"] = state.PlayerActionsDisabled,
            ["is_action_queue_running"] = state.IsActionQueueRunning,
            ["is_hand_selection_active"] = state.IsHandSelectionActive,
            ["can_end_turn"] = state.CanEndTurn,
            ["player"] = state.Player != null ? ShapeCombatEnvPlayer(state.Player) : null,
            ["enemies"] = state.Enemies.Select(ShapeCombatEnvEnemy).ToList(),
            ["hand"] = state.Hand.Select(ShapeCombatEnvHandCard).ToList(),
            ["hand_selection"] = state.HandSelection != null ? ShapeCombatEnvHandSelection(state.HandSelection) : null,
            ["piles"] = new Dictionary<string, object?>
            {
                ["draw"] = state.Piles.Draw,
                ["discard"] = state.Piles.Discard,
                ["exhaust"] = state.Piles.Exhaust,
                ["play"] = state.Piles.Play
            }
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvHandSelection(CombatTrainingHandSelectionSnapshot handSelection)
    {
        return new Dictionary<string, object?>
        {
            ["mode"] = handSelection.Mode.ToLowerInvariant(),
            ["prompt_text"] = handSelection.PromptText,
            ["min_select"] = handSelection.MinSelect,
            ["max_select"] = handSelection.MaxSelect,
            ["can_confirm"] = handSelection.CanConfirm,
            ["cancelable"] = handSelection.Cancelable,
            ["selectable_cards"] = handSelection.SelectableCards.Select(ShapeCombatEnvHandCard).ToList(),
            ["selected_cards"] = handSelection.SelectedCards.Select(ShapeCombatEnvHandCard).ToList()
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvPlayer(CombatTrainingPlayerSnapshot player)
    {
        return new Dictionary<string, object?>
        {
            ["net_id"] = player.NetId,
            ["combat_id"] = player.CombatId,
            ["current_hp"] = player.CurrentHp,
            ["max_hp"] = player.MaxHp,
            ["block"] = player.Block,
            ["energy"] = player.Energy,
            ["max_energy"] = player.MaxEnergy,
            ["stars"] = player.Stars,
            ["powers"] = player.Powers.Select(ShapeCombatEnvPower).ToList()
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvEnemy(CombatTrainingCreatureSnapshot enemy)
    {
        return new Dictionary<string, object?>
        {
            ["combat_id"] = enemy.CombatId,
            ["id"] = enemy.Id,
            ["name"] = enemy.Name,
            ["current_hp"] = enemy.CurrentHp,
            ["max_hp"] = enemy.MaxHp,
            ["block"] = enemy.Block,
            ["is_alive"] = enemy.IsAlive,
            ["is_hittable"] = enemy.IsHittable,
            ["next_move_id"] = enemy.NextMoveId,
            ["intends_to_attack"] = enemy.IntendsToAttack,
            ["intents"] = enemy.Intents.Select(ShapeCombatEnvIntent).ToList(),
            ["powers"] = enemy.Powers.Select(ShapeCombatEnvPower).ToList()
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvIntent(CombatTrainingIntentSnapshot intent)
    {
        return new Dictionary<string, object?>
        {
            ["intent_type"] = intent.IntentType.ToLowerInvariant(),
            ["repeats"] = intent.Repeats,
            ["damage"] = intent.Damage,
            ["total_damage"] = intent.TotalDamage
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvPower(CombatTrainingPowerSnapshot power)
    {
        return new Dictionary<string, object?>
        {
            ["id"] = power.Id,
            ["amount"] = power.Amount
        };
    }

    private static Dictionary<string, object?> ShapeCombatEnvHandCard(CombatTrainingHandCardSnapshot card)
    {
        return new Dictionary<string, object?>
        {
            ["hand_index"] = card.HandIndex,
            ["combat_card_index"] = card.CombatCardIndex,
            ["id"] = card.Id,
            ["title"] = card.Title,
            ["energy_cost"] = card.EnergyCost,
            ["costs_x"] = card.CostsX,
            ["star_cost"] = card.StarCost,
            ["target_type"] = ShapeTargetType(card.TargetType),
            ["can_play"] = card.CanPlay,
            ["requires_target"] = card.RequiresTarget,
            ["valid_target_ids"] = card.ValidTargetIds
        };
    }

    private static string ShapeTargetType(TargetType targetType)
    {
        return targetType switch
        {
            TargetType.None => "none",
            TargetType.Self => "self",
            TargetType.AnyEnemy => "any_enemy",
            TargetType.AnyPlayer => "any_player",
            TargetType.AnyAlly => "any_ally",
            TargetType.TargetedNoCreature => "targeted_no_creature",
            _ => targetType.ToString().ToLowerInvariant()
        };
    }
}
