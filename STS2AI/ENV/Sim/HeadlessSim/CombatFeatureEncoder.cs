using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Multiplayer;
using MegaCrit.Sts2.Core.Multiplayer.Serialization;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace HeadlessSim;

internal sealed class CombatModelMetadata
{
	public const int CombatScalarDim = 18;
	public const int CombatExtraScalarDim = 14;
	public const int MaxHandSize = 12;
	public const int MaxEnemies = 5;
	public const int MaxActions = 30;
	public const int CardAuxDim = 53;
	public const int EnemyAuxDim = 40;
	public const int MaxDeckSize = 50;
	public const int MaxPileSize = 30;

	public bool HasValueOutput { get; init; }

	public bool HasContinuationOutput { get; init; }

	public bool HasDeckInputs { get; init; }

	public bool HasExtraScalarsInput { get; init; }

	public bool HasPileInputs { get; init; }

	public string PolicyOutputName { get; init; } = "policy_logits";

	public string? ValueOutputName { get; init; }

	public string? ContinuationOutputName { get; init; }
}

internal sealed class CombatEncodedFeatures
{
	public float[] Scalars { get; init; } = Array.Empty<float>();

	public float[] ExtraScalars { get; init; } = Array.Empty<float>();

	public long[] HandIds { get; init; } = Array.Empty<long>();

	public float[] HandAux { get; init; } = Array.Empty<float>();

	public float[] HandMask { get; init; } = Array.Empty<float>();

	public long[] EnemyIds { get; init; } = Array.Empty<long>();

	public float[] EnemyAux { get; init; } = Array.Empty<float>();

	public float[] EnemyMask { get; init; } = Array.Empty<float>();

	public long[] ActionTypeIds { get; init; } = Array.Empty<long>();

	public long[] TargetCardIds { get; init; } = Array.Empty<long>();

	public long[] TargetEnemyIds { get; init; } = Array.Empty<long>();

	public float[] ActionMask { get; init; } = Array.Empty<float>();

	public long[]? DeckIds { get; set; }

	public float[]? DeckAux { get; set; }

	public float[]? DeckMask { get; set; }

	public long[]? DrawPileIds { get; set; }

	public float[]? DrawPileAux { get; set; }

	public float[]? DrawPileMask { get; set; }

	public long[]? DiscardPileIds { get; set; }

	public float[]? DiscardPileAux { get; set; }

	public float[]? DiscardPileMask { get; set; }

	public long[]? ExhaustPileIds { get; set; }

	public float[]? ExhaustPileAux { get; set; }

	public float[]? ExhaustPileMask { get; set; }
}

internal sealed class CombatVocab
{
	private sealed class CardProps
	{
		public int TypeIdx { get; init; } = 6;

		public int RarityIdx { get; init; } = 9;
	}

	private static readonly string[] FunctionalTags =
	{
		"self_target", "single_target", "aoe", "random_target",
		"damage", "block", "multi_hit", "x_cost",
		"draw", "energy_gen", "hp_loss", "heal",
		"strength", "dexterity", "strength_scaling", "block_scaling",
		"vulnerable", "weak", "poison", "frail",
		"exhaust", "ethereal", "innate", "retain", "sly",
		"discard", "exhaust_other", "generate_card", "upgrade_card",
		"strike_tag", "defend_tag", "shiv_tag",
		"summon", "forge",
	};

	private readonly Dictionary<string, int> _cardToIdx = new(StringComparer.OrdinalIgnoreCase);
	private readonly Dictionary<string, int> _monsterToIdx = new(StringComparer.OrdinalIgnoreCase);
	private readonly Dictionary<string, HashSet<string>> _cardTags = new(StringComparer.OrdinalIgnoreCase);
	private readonly List<CardProps> _cardProps = new();
	private readonly Dictionary<string, int> _functionalTagToIdx;

	private CombatVocab()
	{
		_functionalTagToIdx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
		for (int i = 0; i < FunctionalTags.Length; i++)
		{
			_functionalTagToIdx[FunctionalTags[i]] = i;
		}
	}

	public static CombatVocab Load(string onnxPath, string? vocabPath)
	{
		CombatVocab vocab = new CombatVocab();
		string? repoVocabPath = ResolvePythonDataFile("vocab.json");
		string? repoCardTagsPath = ResolvePythonDataFile("card_tags.json");

		List<string> vocabCandidates = new();
		if (!string.IsNullOrWhiteSpace(vocabPath))
		{
			vocabCandidates.Add(Path.GetFullPath(vocabPath));
		}
		if (!string.IsNullOrWhiteSpace(repoVocabPath))
		{
			vocabCandidates.Add(repoVocabPath);
		}

		foreach (string candidate in vocabCandidates.Distinct(StringComparer.OrdinalIgnoreCase))
		{
			if (File.Exists(candidate))
			{
				vocab.LoadVocabJson(candidate);
			}
		}

		if (!string.IsNullOrWhiteSpace(repoCardTagsPath) && File.Exists(repoCardTagsPath))
		{
			vocab.LoadCardTags(repoCardTagsPath);
		}

		if (vocab._cardToIdx.Count == 0 || vocab._monsterToIdx.Count == 0)
		{
			throw new InvalidOperationException(
				$"Failed to load combat vocab for ONNX model '{onnxPath}'. " +
				"Expected vocab_mapping.json next to the model or STS2AI/Python/vocab.json in the repo.");
		}

		return vocab;
	}

	public int CardIdx(string? cardId)
	{
		string normalized = NormalizeCardId(cardId);
		return _cardToIdx.TryGetValue(normalized, out int idx) ? idx : 1;
	}

	public int CardIdxPythonPileString(string? cardId)
	{
		string raw = (cardId ?? string.Empty).Trim();
		if (raw.Length == 0)
		{
			return 0;
		}

		// Match the current Python combat encoder exactly: pile string ids are
		// looked up without slugging/lowercasing, so uppercase pipe ids like
		// "DEFEND_IRONCLAD" fall through to 0 even though hand/deck card ids do not.
		if (!string.Equals(raw, raw.ToLowerInvariant(), StringComparison.Ordinal))
		{
			return 0;
		}

		return _cardToIdx.TryGetValue(raw, out int idx) ? idx : 0;
	}

	public int MonsterIdx(string? monsterId)
	{
		string normalized = NormalizeMonsterId(monsterId);
		return _monsterToIdx.TryGetValue(normalized, out int idx) ? idx : 1;
	}

	public int CardTypeIdx(int cardIdx)
	{
		return cardIdx >= 0 && cardIdx < _cardProps.Count ? _cardProps[cardIdx].TypeIdx : 6;
	}

	public int CardRarityIdx(int cardIdx)
	{
		return cardIdx >= 0 && cardIdx < _cardProps.Count ? _cardProps[cardIdx].RarityIdx : 9;
	}

	public void ApplyCardTags(string? cardId, Span<float> destination)
	{
		if (string.IsNullOrWhiteSpace(cardId))
		{
			return;
		}

		if (!_cardTags.TryGetValue(NormalizeCardId(cardId), out HashSet<string>? tags))
		{
			return;
		}

		foreach (string tag in tags)
		{
			if (_functionalTagToIdx.TryGetValue(tag, out int index) && index >= 0 && index < destination.Length)
			{
				destination[index] = 1f;
			}
		}
	}

	private void LoadVocabJson(string path)
	{
		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
		JsonElement root = document.RootElement;
		LoadStringIntMap(root, "card_to_idx", _cardToIdx);
		LoadStringIntMap(root, "monster_to_idx", _monsterToIdx);
		if (root.TryGetProperty("card_props", out JsonElement cardPropsElement) && cardPropsElement.ValueKind == JsonValueKind.Array)
		{
			_cardProps.Clear();
			foreach (JsonElement entry in cardPropsElement.EnumerateArray())
			{
				_cardProps.Add(new CardProps
				{
					TypeIdx = entry.TryGetProperty("type_idx", out JsonElement typeIdxElement) ? typeIdxElement.GetInt32() : 6,
					RarityIdx = entry.TryGetProperty("rarity_idx", out JsonElement rarityIdxElement) ? rarityIdxElement.GetInt32() : 9
				});
			}
		}
	}

	private void LoadCardTags(string path)
	{
		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
		JsonElement root = document.RootElement;
		JsonElement source = root;
		if (root.TryGetProperty("card_tags", out JsonElement nestedCardTags) && nestedCardTags.ValueKind == JsonValueKind.Object)
		{
			source = nestedCardTags;
		}

		foreach (JsonProperty property in source.EnumerateObject())
		{
			if (property.Value.ValueKind != JsonValueKind.Array)
			{
				continue;
			}

			HashSet<string> tags = new(StringComparer.OrdinalIgnoreCase);
			foreach (JsonElement tagElement in property.Value.EnumerateArray())
			{
				string? tag = tagElement.GetString();
				if (!string.IsNullOrWhiteSpace(tag))
				{
					tags.Add(tag.Trim().ToLowerInvariant());
				}
			}

			if (tags.Count > 0)
			{
				_cardTags[NormalizeCardId(property.Name)] = tags;
			}
		}
	}

	private static void LoadStringIntMap(JsonElement root, string propertyName, Dictionary<string, int> destination)
	{
		if (!root.TryGetProperty(propertyName, out JsonElement mapElement) || mapElement.ValueKind != JsonValueKind.Object)
		{
			return;
		}

		foreach (JsonProperty entry in mapElement.EnumerateObject())
		{
			destination[entry.Name.Trim().ToLowerInvariant()] = entry.Value.GetInt32();
		}
	}

	private static string NormalizeCardId(string? cardId)
	{
		return (cardId ?? string.Empty).Trim().ToLowerInvariant();
	}

	private static string NormalizeMonsterId(string? monsterId)
	{
		string value = (monsterId ?? string.Empty).Trim().ToLowerInvariant();
		if (value.Length == 0)
		{
			return value;
		}

		int suffixSep = Math.Max(value.LastIndexOf('_'), value.LastIndexOf('-'));
		if (suffixSep > 0 && suffixSep + 1 < value.Length)
		{
			bool numericSuffix = true;
			for (int i = suffixSep + 1; i < value.Length; i++)
			{
				if (!char.IsDigit(value[i]))
				{
					numericSuffix = false;
					break;
				}
			}

			if (numericSuffix)
			{
				return value[..suffixSep];
			}
		}

		return value;
	}

	private static string? ResolvePythonDataFile(string fileName)
	{
		DirectoryInfo? cursor = new DirectoryInfo(AppContext.BaseDirectory);
		while (cursor != null)
		{
			string candidate = Path.Combine(cursor.FullName, "STS2AI", "Python", fileName);
			if (File.Exists(candidate))
			{
				return candidate;
			}

			cursor = cursor.Parent;
		}

		return null;
	}
}

internal sealed class CombatFeatureEncoder
{
	private static readonly Dictionary<string, int> CardTypeMap = new(StringComparer.OrdinalIgnoreCase)
	{
		["attack"] = 0,
		["skill"] = 1,
		["power"] = 2,
		["status"] = 3,
		["curse"] = 4,
		["quest"] = 5,
		["none"] = 6,
	};

	private static readonly Dictionary<string, int> CardRarityMap = new(StringComparer.OrdinalIgnoreCase)
	{
		["basic"] = 0,
		["common"] = 1,
		["uncommon"] = 2,
		["rare"] = 3,
		["ancient"] = 4,
		["event"] = 5,
		["token"] = 6,
		["status"] = 7,
		["curse"] = 8,
		["none"] = 9,
	};

	private static readonly Dictionary<string, int> CombatActionTypeMap = new(StringComparer.OrdinalIgnoreCase)
	{
		["play_card"] = 0,
		["end_turn"] = 1,
		["use_potion"] = 2,
		["select_hand_card"] = 3,
		["select_card_option"] = 4,
		["confirm_selection"] = 5,
		["cancel_selection"] = 6,
		["other"] = 7,
	};

	private readonly CombatVocab _vocab;

	public CombatFeatureEncoder(CombatVocab vocab)
	{
		_vocab = vocab;
	}

	public CombatEncodedFeatures Encode(
		FullRunSimulationStateSnapshot snapshot,
		IReadOnlyList<FullRunSimulationLegalAction> legalActions,
		CombatModelMetadata metadata)
	{
		CombatTrainingStateSnapshot? combat = snapshot.CachedCombatState;
		CombatTrainingPlayerSnapshot? combatPlayer = combat?.Player;
		Player? localPlayer = TryResolveLocalPlayer(RunManager.Instance.DebugOnlyGetState());

		float[] scalars = new float[CombatModelMetadata.CombatScalarDim];
		float[] extraScalars = new float[CombatModelMetadata.CombatExtraScalarDim];
		long[] handIds = new long[CombatModelMetadata.MaxHandSize];
		float[] handAux = new float[CombatModelMetadata.MaxHandSize * CombatModelMetadata.CardAuxDim];
		float[] handMask = new float[CombatModelMetadata.MaxHandSize];
		long[] enemyIds = new long[CombatModelMetadata.MaxEnemies];
		float[] enemyAux = new float[CombatModelMetadata.MaxEnemies * CombatModelMetadata.EnemyAuxDim];
		float[] enemyMask = new float[CombatModelMetadata.MaxEnemies];
		long[] actionTypeIds = new long[CombatModelMetadata.MaxActions];
		long[] targetCardIds = new long[CombatModelMetadata.MaxActions];
		long[] targetEnemyIds = new long[CombatModelMetadata.MaxActions];
		float[] actionMask = new float[CombatModelMetadata.MaxActions];

		BuildScalarFeatures(snapshot, combatPlayer, combat, localPlayer, scalars, extraScalars);
		BuildHandFeatures(combat, handIds, handAux, handMask);
		BuildEnemyFeatures(combat, enemyIds, enemyAux, enemyMask, extraScalars);
		BuildActionFeatures(combat, legalActions, actionTypeIds, targetCardIds, targetEnemyIds, actionMask);

		CombatEncodedFeatures encoded = new CombatEncodedFeatures
		{
			Scalars = scalars,
			ExtraScalars = extraScalars,
			HandIds = handIds,
			HandAux = handAux,
			HandMask = handMask,
			EnemyIds = enemyIds,
			EnemyAux = enemyAux,
			EnemyMask = enemyMask,
			ActionTypeIds = actionTypeIds,
			TargetCardIds = targetCardIds,
			TargetEnemyIds = targetEnemyIds,
			ActionMask = actionMask,
		};

		if (metadata.HasDeckInputs || metadata.HasPileInputs)
		{
			PopulateDeckAndPileFeatures(encoded, combat, localPlayer, metadata);
		}

		return encoded;
	}

	private void BuildScalarFeatures(
		FullRunSimulationStateSnapshot snapshot,
		CombatTrainingPlayerSnapshot? combatPlayer,
		CombatTrainingStateSnapshot? combat,
		Player? localPlayer,
		float[] scalars,
		float[] extraScalars)
	{
		float hp = combatPlayer?.CurrentHp ?? localPlayer?.Creature.CurrentHp ?? 0;
		float maxHp = Math.Max(1f, combatPlayer?.MaxHp ?? localPlayer?.Creature.MaxHp ?? 1);
		scalars[0] = hp / maxHp;
		scalars[1] = maxHp / 100f;
		scalars[2] = (combatPlayer?.Block ?? localPlayer?.Creature.Block ?? 0) / 50f;
		scalars[3] = (combatPlayer?.Energy ?? 0) / 5f;
		scalars[4] = (combatPlayer?.MaxEnergy ?? localPlayer?.MaxEnergy ?? 3) / 5f;
		// Match Python build_combat_features(): the current API state does not
		// expose round_number on the combat path, so round 1 encodes as 0.
		scalars[5] = Math.Max(0f, (combat?.RoundNumber ?? 0) - 1f) / 20f;
		scalars[6] = (combat?.Piles?.Draw ?? 0) / 30f;
		scalars[7] = (combat?.Piles?.Discard ?? 0) / 30f;
		scalars[8] = (combat?.Piles?.Exhaust ?? 0) / 20f;
		scalars[9] = snapshot.TotalFloor / 20f;

		IEnumerable<CombatTrainingPowerSnapshot> playerPowers = combatPlayer?.Powers?.AsEnumerable() ?? Array.Empty<CombatTrainingPowerSnapshot>();
		scalars[10] = GetPowerAmount(playerPowers, "strength") / 10f;
		scalars[11] = GetPowerAmount(playerPowers, "dexterity") / 10f;
		scalars[12] = Math.Min(GetPowerAmount(playerPowers, "vulnerable") / 5f, 1f);
		scalars[13] = Math.Min(GetPowerAmount(playerPowers, "weak") / 5f, 1f);
		scalars[14] = Math.Min(GetPowerAmount(playerPowers, "frail") / 5f, 1f);
		scalars[15] = GetPowerAmount(playerPowers, "metallicize") / 10f;
		scalars[16] = GetPowerAmount(playerPowers, "regen") / 10f;
		scalars[17] = Math.Min(GetPowerAmount(playerPowers, "artifact") / 3f, 1f);

		extraScalars[0] = Math.Min(GetPowerAmount(playerPowers, "intangible") / 5f, 1f);
		extraScalars[1] = Math.Min(GetPowerAmount(playerPowers, "barricade") / 1f, 1f);
		extraScalars[2] = GetPowerAmount(playerPowers, "inflame") / 10f;
		extraScalars[3] = Math.Min(GetPowerAmount(playerPowers, "demon_form") / 5f, 1f);
		extraScalars[4] = Math.Min(GetPowerAmount(playerPowers, "flame_barrier") / 12f, 1f);
		extraScalars[5] = GetPowerAmount(playerPowers, "thorns") / 10f;
		extraScalars[6] = GetPowerAmount(playerPowers, "plated_armor") / 30f;
		extraScalars[7] = Math.Min(GetPowerAmount(playerPowers, "double_tap") / 3f, 1f);
		extraScalars[8] = Math.Min(GetPowerAmount(playerPowers, "energized") / 5f, 1f);
		extraScalars[9] = Math.Min(GetPowerAmount(playerPowers, "feel_no_pain") / 10f, 1f);
		extraScalars[10] = Math.Min(GetPowerAmount(playerPowers, "dark_embrace") / 1f, 1f);
		extraScalars[11] = Math.Min(GetPowerAmount(playerPowers, "evolve") / 3f, 1f);
		extraScalars[12] = Math.Min(GetPowerAmount(playerPowers, "strength_up") / 3f, 1f);
	}

	private void BuildHandFeatures(CombatTrainingStateSnapshot? combat, long[] handIds, float[] handAux, float[] handMask)
	{
		if (combat == null)
		{
			return;
		}

		for (int i = 0; i < Math.Min(combat.Hand.Count, CombatModelMetadata.MaxHandSize); i++)
		{
			CombatTrainingHandCardSnapshot card = combat.Hand[i];
			int cardIdx = _vocab.CardIdx(card.Id);
			handIds[i] = cardIdx;
			handMask[i] = 1f;
			BuildCardAux(card, cardIdx, handAux.AsSpan(i * CombatModelMetadata.CardAuxDim, CombatModelMetadata.CardAuxDim));
		}
	}

	private void BuildEnemyFeatures(CombatTrainingStateSnapshot? combat, long[] enemyIds, float[] enemyAux, float[] enemyMask, float[] extraScalars)
	{
		if (combat == null)
		{
			return;
		}

		int aliveCount = 0;
		int minionCount = 0;
		foreach (CombatTrainingCreatureSnapshot enemy in combat.Enemies)
		{
			if (!enemy.IsAlive || aliveCount >= CombatModelMetadata.MaxEnemies)
			{
				continue;
			}

			enemyIds[aliveCount] = _vocab.MonsterIdx(enemy.Id);
			enemyMask[aliveCount] = 1f;
			bool isMinion = BuildEnemyAux(enemy, enemyAux.AsSpan(aliveCount * CombatModelMetadata.EnemyAuxDim, CombatModelMetadata.EnemyAuxDim));
			if (isMinion)
			{
				minionCount++;
			}
			aliveCount++;
		}

		extraScalars[13] = aliveCount == 0 ? 0f : (float)minionCount / aliveCount;
	}

	private void BuildActionFeatures(
		CombatTrainingStateSnapshot? combat,
		IReadOnlyList<FullRunSimulationLegalAction> legalActions,
		long[] actionTypeIds,
		long[] targetCardIds,
		long[] targetEnemyIds,
		float[] actionMask)
	{
		for (int i = 0; i < Math.Min(legalActions.Count, CombatModelMetadata.MaxActions); i++)
		{
			FullRunSimulationLegalAction action = legalActions[i];
			string actionName = Normalize(action.Action);
			actionMask[i] = 1f;
			actionTypeIds[i] = CombatActionTypeMap.TryGetValue(actionName, out int actionType)
				? actionType
				: CombatActionTypeMap["other"];

			if (!string.Equals(actionName, "play_card", StringComparison.Ordinal) || combat == null)
			{
				continue;
			}

			int cardIndex = action.CardIndex ?? action.Index ?? -1;
			if (cardIndex >= 0 && cardIndex < combat.Hand.Count)
			{
				targetCardIds[i] = _vocab.CardIdx(combat.Hand[cardIndex].Id);
			}
			else if (!string.IsNullOrWhiteSpace(action.CardId))
			{
				targetCardIds[i] = _vocab.CardIdx(action.CardId);
			}

			if (!action.TargetId.HasValue)
			{
				continue;
			}

			// Match Python build_combat_action_features(): on the current pipe
			// state it compares target_id against entity_id and zero-based alive
			// slot index, not combat_id. That means many targeted combat actions
			// intentionally encode target_enemy_ids=0 in today's baseline.
			int targetId = unchecked((int)action.TargetId.Value);
			int aliveEnemyIndex = 0;
			foreach (CombatTrainingCreatureSnapshot enemy in combat.Enemies)
			{
				if (!enemy.IsAlive)
				{
					continue;
				}

				if (aliveEnemyIndex == targetId)
				{
					targetEnemyIds[i] = _vocab.MonsterIdx(enemy.Id);
					break;
				}

				aliveEnemyIndex++;
			}
		}
	}

	private void PopulateDeckAndPileFeatures(
		CombatEncodedFeatures encoded,
		CombatTrainingStateSnapshot? combat,
		Player? localPlayer,
		CombatModelMetadata metadata)
	{
		if (metadata.HasDeckInputs)
		{
			long[] deckIds = new long[CombatModelMetadata.MaxDeckSize];
			float[] deckAux = new float[CombatModelMetadata.MaxDeckSize * CombatModelMetadata.CardAuxDim];
			float[] deckMask = new float[CombatModelMetadata.MaxDeckSize];

			if (localPlayer != null)
			{
				int count = 0;
				foreach (CardModel card in localPlayer.Deck.Cards)
				{
					if (count >= CombatModelMetadata.MaxDeckSize)
					{
						break;
					}

					int cardIdx = _vocab.CardIdx(card.Id.Entry);
					deckIds[count] = cardIdx;
					deckMask[count] = 1f;
					BuildCardAux(card, cardIdx, deckAux.AsSpan(count * CombatModelMetadata.CardAuxDim, CombatModelMetadata.CardAuxDim));
					count++;
				}
			}

			encoded.DeckIds = deckIds;
			encoded.DeckAux = deckAux;
			encoded.DeckMask = deckMask;
		}

		if (metadata.HasPileInputs)
		{
			encoded.DrawPileIds = new long[CombatModelMetadata.MaxPileSize];
			encoded.DrawPileAux = new float[CombatModelMetadata.MaxPileSize * CombatModelMetadata.CardAuxDim];
			encoded.DrawPileMask = new float[CombatModelMetadata.MaxPileSize];
			encoded.DiscardPileIds = new long[CombatModelMetadata.MaxPileSize];
			encoded.DiscardPileAux = new float[CombatModelMetadata.MaxPileSize * CombatModelMetadata.CardAuxDim];
			encoded.DiscardPileMask = new float[CombatModelMetadata.MaxPileSize];
			encoded.ExhaustPileIds = new long[CombatModelMetadata.MaxPileSize];
			encoded.ExhaustPileAux = new float[CombatModelMetadata.MaxPileSize * CombatModelMetadata.CardAuxDim];
			encoded.ExhaustPileMask = new float[CombatModelMetadata.MaxPileSize];

			EncodePileCardIds(combat?.Piles.DrawCardIds, encoded.DrawPileIds, encoded.DrawPileMask);
			EncodePileCardIds(combat?.Piles.DiscardCardIds, encoded.DiscardPileIds, encoded.DiscardPileMask);
			EncodePileCardIds(combat?.Piles.ExhaustCardIds, encoded.ExhaustPileIds, encoded.ExhaustPileMask);
		}
	}

	private void EncodePileCardIds(IReadOnlyList<string>? cardIds, long[] destinationIds, float[] destinationMask)
	{
		if (cardIds == null)
		{
			return;
		}

		for (int i = 0; i < Math.Min(cardIds.Count, CombatModelMetadata.MaxPileSize); i++)
		{
			string? cardId = cardIds[i];
			if (string.IsNullOrWhiteSpace(cardId))
			{
				continue;
			}

			destinationIds[i] = _vocab.CardIdxPythonPileString(cardId);
			destinationMask[i] = 1f;
		}
	}

	private void BuildCardAux(CombatTrainingHandCardSnapshot card, int cardIdx, Span<float> destination)
	{
		destination.Clear();
		destination[0] = card.EnergyCost / 5f;
		int typeIdx = CardTypeMap.TryGetValue(Normalize(card.CardType), out int parsedTypeIdx) ? parsedTypeIdx : _vocab.CardTypeIdx(cardIdx);
		int rarityIdx = _vocab.CardRarityIdx(cardIdx);
		destination[1 + Math.Clamp(typeIdx, 0, 6)] = 1f;
		destination[8 + Math.Clamp(rarityIdx, 0, 9)] = 1f;
		destination[18] = card.IsUpgraded ? 1f : 0f;
		_vocab.ApplyCardTags(card.Id, destination[19..]);
	}

	private void BuildCardAux(CardModel card, int cardIdx, Span<float> destination)
	{
		destination.Clear();
		destination[0] = card.EnergyCost.GetWithModifiers(CostModifiers.All) / 5f;
		int typeIdx = CardTypeMap.TryGetValue(Normalize(card.Type.ToString()), out int parsedTypeIdx) ? parsedTypeIdx : _vocab.CardTypeIdx(cardIdx);
		int rarityIdx = CardRarityMap.TryGetValue(Normalize(card.Rarity.ToString()), out int parsedRarityIdx) ? parsedRarityIdx : _vocab.CardRarityIdx(cardIdx);
		destination[1 + Math.Clamp(typeIdx, 0, 6)] = 1f;
		destination[8 + Math.Clamp(rarityIdx, 0, 9)] = 1f;
		destination[18] = card.IsUpgraded ? 1f : 0f;
		_vocab.ApplyCardTags(card.Id.Entry, destination[19..]);
	}

	private static bool BuildEnemyAux(CombatTrainingCreatureSnapshot enemy, Span<float> destination)
	{
		destination.Clear();
		float currentHp = enemy.CurrentHp;
		float maxHp = Math.Max(1f, enemy.MaxHp);
		destination[0] = currentHp / maxHp;
		destination[1] = maxHp / 200f;
		destination[2] = enemy.Block / 50f;

		List<CombatTrainingIntentSnapshot> intents = enemy.Intents ?? new List<CombatTrainingIntentSnapshot>();
		string intentText = string.Join(" ", intents.Select(static intent => Normalize(intent.IntentType)));
		CombatTrainingIntentSnapshot? primaryIntent = intents.FirstOrDefault(static intent =>
			(intent.Damage ?? intent.TotalDamage ?? 0) > 0) ?? intents.FirstOrDefault();
		float perHitDamage = primaryIntent?.Damage ?? primaryIntent?.TotalDamage ?? 0;
		float repeats = Math.Max(1, primaryIntent?.Repeats ?? 1);
		float totalIntentDamage = intents.Sum(static intent => (float)(intent.TotalDamage ?? intent.Damage ?? 0));

		destination[3] = intentText.Contains("attack", StringComparison.Ordinal) ? 1f : 0f;
		destination[4] = (intentText.Contains("defend", StringComparison.Ordinal) || intentText.Contains("block", StringComparison.Ordinal)) ? 1f : 0f;
		destination[5] = intentText.Contains("buff", StringComparison.Ordinal) && !intentText.Contains("debuff", StringComparison.Ordinal) ? 1f : 0f;
		destination[6] = intentText.Contains("debuff", StringComparison.Ordinal) ? 1f : 0f;
		destination[7] = perHitDamage / 30f;
		destination[8] = repeats / 5f;
		destination[9] = maxHp >= 80f ? 1f : 0f;

		destination[10] = GetPowerAmount(enemy.Powers, "strength") / 10f;
		destination[11] = Math.Min(GetPowerAmount(enemy.Powers, "vulnerable") / 5f, 1f);
		destination[12] = Math.Min(GetPowerAmount(enemy.Powers, "weak") / 5f, 1f);
		destination[13] = Math.Min(GetPowerAmount(enemy.Powers, "poison") / 20f, 1f);
		destination[14] = Math.Min(GetPowerAmount(enemy.Powers, "artifact") / 3f, 1f);
		destination[15] = GetPowerAmount(enemy.Powers, "regen") / 10f;

		bool isMinion = GetPowerAmount(enemy.Powers, "minion") > 0f;
		destination[16] = Math.Min(GetPowerAmount(enemy.Powers, "slippery") / 9f, 1f);
		destination[17] = Math.Min(GetPowerAmount(enemy.Powers, "intangible") / 5f, 1f);
		destination[18] = Math.Min(GetPowerAmount(enemy.Powers, "hardtokill") / 5f, 1f);
		destination[19] = isMinion ? 1f : 0f;
		destination[20] = Math.Min(GetPowerAmount(enemy.Powers, "metallicize") / 10f, 1f);
		destination[21] = Math.Min(GetPowerAmount(enemy.Powers, "barricade") / 1f, 1f);
		destination[22] = Math.Min(GetPowerAmount(enemy.Powers, "ritual") / 5f, 1f);
		destination[23] = Math.Min(GetPowerAmount(enemy.Powers, "angry") / 5f, 1f);
		destination[24] = Math.Min(GetPowerAmount(enemy.Powers, "curl_up") / 30f, 1f);
		destination[25] = Math.Min(GetPowerAmount(enemy.Powers, "thorns") / 10f, 1f);
		destination[26] = Math.Min(GetPowerAmount(enemy.Powers, "plated_armor") / 30f, 1f);
		destination[27] = Math.Min(GetPowerAmount(enemy.Powers, "plating") / 10f, 1f);
		destination[28] = Math.Min(GetPowerAmount(enemy.Powers, "hardenedshell") / 10f, 1f);
		destination[29] = Math.Min(GetPowerAmount(enemy.Powers, "enrage") / 5f, 1f);
		destination[30] = Math.Min(GetPowerAmount(enemy.Powers, "mode_shift") / 30f, 1f);
		destination[31] = Math.Min(GetPowerAmount(enemy.Powers, "flight") / 5f, 1f);
		destination[32] = Math.Min(GetPowerAmount(enemy.Powers, "spore_cloud") / 5f, 1f);
		destination[33] = Math.Min(GetPowerAmount(enemy.Powers, "plow") / 5f, 1f);
		destination[34] = enemy.IsHittable ? 1f : 0f;
		destination[35] = enemy.IntendsToAttack ? 1f : destination[3];
		destination[36] = HashMoveId(enemy.NextMoveId);
		destination[37] = Math.Min(intents.Count / 4f, 1f);
		destination[38] = Math.Min(totalIntentDamage / 30f, 2f);
		return isMinion;
	}

	private static float GetPowerAmount(IEnumerable<CombatTrainingPowerSnapshot>? powers, string powerId)
	{
		if (powers == null)
		{
			return 0f;
		}

		foreach (CombatTrainingPowerSnapshot power in powers)
		{
			if (Normalize(power.Id).Contains(powerId, StringComparison.Ordinal))
			{
				return power.Amount;
			}
		}

		return 0f;
	}

	private static float HashMoveId(string? nextMoveId)
	{
		if (string.IsNullOrWhiteSpace(nextMoveId))
		{
			return 0f;
		}

		byte[] digest = System.Security.Cryptography.MD5.HashData(System.Text.Encoding.UTF8.GetBytes(nextMoveId));
		int value = (digest[0] << 8) | digest[1];
		return (value % 65536) / 65536f;
	}

	private static Player? TryResolveLocalPlayer(RunState? runState)
	{
		if (runState == null)
		{
			return null;
		}

		return runState.Players.FirstOrDefault(static player => player.NetId == NetSingleplayerGameService.defaultNetId)
			?? runState.Players.FirstOrDefault();
	}

	private static string Normalize(string? value)
	{
		return (value ?? string.Empty).Trim().ToLowerInvariant();
	}
}
