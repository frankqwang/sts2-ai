using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MegaCrit.Sts2.Core.Simulation;
using MegaCrit.Sts2.Core.Training;

namespace HeadlessSim;

/// <summary>
/// Local ONNX Runtime CPU actor policy for combat inference.
/// Replaces Python GPU inference on the rollout hot path.
///
/// Usage:
///   var policy = new OrtActorPolicy("actor_combat.onnx");
///   var (actionIdx, logits) = policy.SelectAction(snapshot, session, rng);
///   policy.Dispose();
/// </summary>
internal sealed class OrtActorPolicy : IDisposable
{
	// Feature dimensions (must match Python export_actor_onnx.py exactly)
	private const int COMBAT_SCALAR_DIM = 18;
	private const int MAX_HAND_SIZE = 12;
	private const int MAX_ENEMIES = 5;
	private const int MAX_ACTIONS = 30;
	private const int CARD_AUX_DIM = 53;  // embed_dim(48) + cost(1) + type(1) + rarity(1) + upgraded(1) + target(1)
	private const int ENEMY_AUX_DIM = 40;

	private readonly InferenceSession _session;
	private readonly bool _argmax;
	private readonly Dictionary<string, int> _cardVocab = new(StringComparer.OrdinalIgnoreCase);
	private readonly Dictionary<string, int> _monsterVocab = new(StringComparer.OrdinalIgnoreCase);

	// Pre-allocated buffers to avoid per-step allocation
	private readonly float[] _scalars = new float[COMBAT_SCALAR_DIM];
	private readonly long[] _handIds = new long[MAX_HAND_SIZE];
	private readonly float[] _handAux = new float[MAX_HAND_SIZE * CARD_AUX_DIM];
	private readonly float[] _handMask = new float[MAX_HAND_SIZE];
	private readonly long[] _enemyIds = new long[MAX_ENEMIES];
	private readonly float[] _enemyAux = new float[MAX_ENEMIES * ENEMY_AUX_DIM];
	private readonly float[] _enemyMask = new float[MAX_ENEMIES];
	private readonly long[] _actionTypeIds = new long[MAX_ACTIONS];
	private readonly long[] _targetCardIds = new long[MAX_ACTIONS];
	private readonly long[] _targetEnemyIds = new long[MAX_ACTIONS];
	private readonly float[] _actionMask = new float[MAX_ACTIONS];

	public OrtActorPolicy(string onnxPath, bool argmax = true, string? vocabPath = null)
	{
		var opts = new SessionOptions
		{
			// Single-threaded to avoid oversubscription with 8+ env processes
			IntraOpNumThreads = 1,
			InterOpNumThreads = 1,
			ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
			GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
		};
		_session = new InferenceSession(onnxPath, opts);
		_argmax = argmax;

		// Load vocab mapping (Python vocab indices, not binary session symbols)
		if (vocabPath != null && System.IO.File.Exists(vocabPath))
		{
			try
			{
				string json = System.IO.File.ReadAllText(vocabPath);
				var doc = System.Text.Json.JsonDocument.Parse(json);
				if (doc.RootElement.TryGetProperty("card_to_idx", out var cards))
					foreach (var kv in cards.EnumerateObject())
						_cardVocab[Slugify(kv.Name)] = kv.Value.GetInt32();
				if (doc.RootElement.TryGetProperty("monster_to_idx", out var monsters))
					foreach (var kv in monsters.EnumerateObject())
						_monsterVocab[Slugify(kv.Name)] = kv.Value.GetInt32();
				Console.Error.WriteLine($"[ORT] Vocab loaded: {_cardVocab.Count} cards, {_monsterVocab.Count} monsters");
			}
			catch (Exception ex)
			{
				Console.Error.WriteLine($"[ORT] Vocab load failed: {ex.Message}");
			}
		}

		// Warmup
		ClearBuffers();
		_handMask[0] = 1f;
		_enemyMask[0] = 1f;
		_actionMask[0] = 1f;
		RunInference();
	}

	private static string Slugify(string s)
	{
		return s.Trim().ToLowerInvariant().Replace(" ", "_").Replace("-", "_");
	}

	private long CardIdx(string? cardId)
	{
		if (string.IsNullOrEmpty(cardId)) return 0;
		string slug = Slugify(cardId);
		return _cardVocab.TryGetValue(slug, out int idx) ? idx : 1; // 1 = <unk>
	}

	private long MonsterIdx(string? monsterId)
	{
		if (string.IsNullOrEmpty(monsterId)) return 0;
		string slug = Slugify(monsterId);
		return _monsterVocab.TryGetValue(slug, out int idx) ? idx : 1; // 1 = <unk>
	}

	/// <summary>
	/// Select an action for the current combat state.
	/// Returns (actionIndex, logits array for all MAX_ACTIONS slots).
	/// </summary>
	public (int ActionIndex, float[] Logits) SelectAction(
		FullRunSimulationStateSnapshot snapshot,
		BinarySessionState? session,
		Random rng)
	{
		EncodeFeatures(snapshot);
		float[] logits = RunInference();

		int numLegal = snapshot.LegalActions.Count;
		int actionIdx;

		if (_argmax || rng == null)
		{
			actionIdx = ArgmaxLegal(logits, numLegal);
		}
		else
		{
			actionIdx = SampleFromLogits(logits, numLegal, rng);
		}

		return (Math.Clamp(actionIdx, 0, Math.Max(0, numLegal - 1)), logits);
	}

	private float[] RunInference()
	{
		var inputs = new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("scalars",
				new DenseTensor<float>(_scalars, new[] { 1, COMBAT_SCALAR_DIM })),
			NamedOnnxValue.CreateFromTensor("hand_ids",
				new DenseTensor<long>(_handIds, new[] { 1, MAX_HAND_SIZE })),
			NamedOnnxValue.CreateFromTensor("hand_aux",
				new DenseTensor<float>(_handAux, new[] { 1, MAX_HAND_SIZE, CARD_AUX_DIM })),
			NamedOnnxValue.CreateFromTensor("hand_mask",
				new DenseTensor<float>(_handMask, new[] { 1, MAX_HAND_SIZE })),
			NamedOnnxValue.CreateFromTensor("enemy_ids",
				new DenseTensor<long>(_enemyIds, new[] { 1, MAX_ENEMIES })),
			NamedOnnxValue.CreateFromTensor("enemy_aux",
				new DenseTensor<float>(_enemyAux, new[] { 1, MAX_ENEMIES, ENEMY_AUX_DIM })),
			NamedOnnxValue.CreateFromTensor("enemy_mask",
				new DenseTensor<float>(_enemyMask, new[] { 1, MAX_ENEMIES })),
			NamedOnnxValue.CreateFromTensor("action_type_ids",
				new DenseTensor<long>(_actionTypeIds, new[] { 1, MAX_ACTIONS })),
			NamedOnnxValue.CreateFromTensor("target_card_ids",
				new DenseTensor<long>(_targetCardIds, new[] { 1, MAX_ACTIONS })),
			NamedOnnxValue.CreateFromTensor("target_enemy_ids",
				new DenseTensor<long>(_targetEnemyIds, new[] { 1, MAX_ACTIONS })),
			NamedOnnxValue.CreateFromTensor("action_mask",
				new DenseTensor<float>(_actionMask, new[] { 1, MAX_ACTIONS })),
		};

		using var results = _session.Run(inputs);
		var logitsTensor = results.First().AsTensor<float>();
		float[] logits = new float[MAX_ACTIONS];
		for (int i = 0; i < MAX_ACTIONS; i++)
			logits[i] = logitsTensor[0, i];
		return logits;
	}

	private void ClearBuffers()
	{
		Array.Clear(_scalars);
		Array.Clear(_handIds);
		Array.Clear(_handAux);
		Array.Clear(_handMask);
		Array.Clear(_enemyIds);
		Array.Clear(_enemyAux);
		Array.Clear(_enemyMask);
		Array.Clear(_actionTypeIds);
		Array.Clear(_targetCardIds);
		Array.Clear(_targetEnemyIds);
		Array.Clear(_actionMask);
	}

	/// <summary>
	/// Encode combat state into flat tensors matching Python build_combat_features.
	/// This is a simplified version — focuses on the fields available in BinaryProtocol.
	/// </summary>
	private void EncodeFeatures(FullRunSimulationStateSnapshot snapshot)
	{
		ClearBuffers();

		var combat = snapshot.CachedCombatState;
		if (combat == null) return;

		// Scalars [0-17]
		float hp = combat.Player?.CurrentHp ?? 0;
		float maxHp = combat.Player?.MaxHp ?? 1;
		_scalars[0] = Math.Clamp(hp / Math.Max(maxHp, 1f), 0f, 1f); // hp_ratio
		_scalars[1] = maxHp / 200f; // max_hp_norm
		_scalars[2] = (combat.Player?.Block ?? 0) / 50f; // block
		_scalars[3] = (combat.Player?.Energy ?? 0) / 5f; // energy
		_scalars[4] = (combat.Player?.MaxEnergy ?? 3) / 5f; // max_energy
		_scalars[5] = (combat.RoundNumber) / 20f; // round
		_scalars[6] = (combat.Piles?.Draw ?? 0) / 30f; // draw_pile
		_scalars[7] = (combat.Piles?.Discard ?? 0) / 30f; // discard_pile
		_scalars[8] = (combat.Piles?.Exhaust ?? 0) / 30f; // exhaust_pile
		_scalars[9] = snapshot.TotalFloor / 50f; // floor
		// Powers [10-17] — from combat state
		if (combat.Player?.Powers != null)
		{
			foreach (var power in combat.Player.Powers)
			{
				int idx = PowerIndex(power.Id);
				if (idx >= 0 && idx < 8)
					_scalars[10 + idx] = power.Amount / 10f;
			}
		}

		// Hand cards
		if (combat.Hand != null)
		{
			for (int i = 0; i < Math.Min(combat.Hand.Count, MAX_HAND_SIZE); i++)
			{
				var card = combat.Hand[i];
				_handIds[i] = CardIdx(card.Id);
				_handMask[i] = 1f;
				// Simplified card aux — energy cost normalized
				int auxBase = i * CARD_AUX_DIM;
				_handAux[auxBase] = card.EnergyCost / 5f;
			}
		}

		// Enemies
		if (combat.Enemies != null)
		{
			int eIdx = 0;
			foreach (var enemy in combat.Enemies)
			{
				if (eIdx >= MAX_ENEMIES || !enemy.IsAlive) continue;
				_enemyIds[eIdx] = MonsterIdx(enemy.Id);
				_enemyMask[eIdx] = 1f;
				int auxBase = eIdx * ENEMY_AUX_DIM;
				float eMaxHp = Math.Max(enemy.MaxHp, 1f);
				_enemyAux[auxBase + 0] = Math.Clamp(enemy.CurrentHp / eMaxHp, 0f, 1f);
				_enemyAux[auxBase + 1] = eMaxHp / 200f;
				_enemyAux[auxBase + 2] = enemy.Block / 50f;
				// Intent damage
				if (enemy.Intents != null && enemy.Intents.Count > 0)
				{
					_enemyAux[auxBase + 3] = (enemy.Intents[0].TotalDamage ?? 0) / 50f;
				}
				eIdx++;
			}
		}

		// Legal actions
		for (int i = 0; i < Math.Min(snapshot.LegalActions.Count, MAX_ACTIONS); i++)
		{
			var action = snapshot.LegalActions[i];
			_actionMask[i] = 1f;
			_actionTypeIds[i] = ActionTypeIndex(action.Action);
			if (action.CardIndex.HasValue)
			{
				// Card target
				string? cardId = GetCardIdAtHandIndex(combat, action.CardIndex.Value);
				if (cardId != null)
					_targetCardIds[i] = CardIdx(cardId);
			}
			if (action.TargetId.HasValue)
			{
				string? enemyId = GetEnemyIdByCombatId(combat, (int)action.TargetId.Value);
				if (enemyId != null)
					_targetEnemyIds[i] = MonsterIdx(enemyId);
			}
		}
	}

	private static int PowerIndex(string? powerId)
	{
		if (string.IsNullOrEmpty(powerId)) return -1;
		string lower = powerId.ToLowerInvariant();
		if (lower.Contains("strength")) return 0;
		if (lower.Contains("dexterity")) return 1;
		if (lower.Contains("vulnerable")) return 2;
		if (lower.Contains("weak")) return 3;
		if (lower.Contains("frail")) return 4;
		if (lower.Contains("metallicize")) return 5;
		if (lower.Contains("regen")) return 6;
		if (lower.Contains("artifact")) return 7;
		return -1;
	}

	private static long ActionTypeIndex(string? action)
	{
		return action?.ToLowerInvariant() switch
		{
			"play_card" => 0,
			"end_turn" => 1,
			"use_potion" or "drink_potion" => 2,
			"select_hand_card" or "combat_select_card" => 3,
			"select_card_option" => 4,
			"confirm_selection" or "combat_confirm_selection" => 5,
			"cancel_selection" => 6,
			_ => 7,
		};
	}

	private static string? GetCardIdAtHandIndex(CombatTrainingStateSnapshot combat, int handIndex)
	{
		if (combat.Hand == null || handIndex < 0 || handIndex >= combat.Hand.Count)
			return null;
		return combat.Hand[handIndex].Id;
	}

	private static string? GetEnemyIdByCombatId(CombatTrainingStateSnapshot combat, int combatId)
	{
		if (combat.Enemies == null) return null;
		foreach (var enemy in combat.Enemies)
		{
			if (enemy.CombatId == (uint)combatId)
				return enemy.Id;
		}
		return null;
	}

	private static int ArgmaxLegal(float[] logits, int numLegal)
	{
		int best = 0;
		float bestVal = float.NegativeInfinity;
		for (int i = 0; i < Math.Min(numLegal, MAX_ACTIONS); i++)
		{
			if (logits[i] > bestVal)
			{
				bestVal = logits[i];
				best = i;
			}
		}
		return best;
	}

	private static int SampleFromLogits(float[] logits, int numLegal, Random rng)
	{
		int n = Math.Min(numLegal, MAX_ACTIONS);
		// Softmax over legal actions
		float maxLogit = float.NegativeInfinity;
		for (int i = 0; i < n; i++)
			maxLogit = Math.Max(maxLogit, logits[i]);

		float sumExp = 0f;
		Span<float> probs = stackalloc float[n];
		for (int i = 0; i < n; i++)
		{
			probs[i] = MathF.Exp(logits[i] - maxLogit);
			sumExp += probs[i];
		}
		for (int i = 0; i < n; i++)
			probs[i] /= sumExp;

		// Sample from categorical
		float u = (float)rng.NextDouble();
		float cumSum = 0f;
		for (int i = 0; i < n; i++)
		{
			cumSum += probs[i];
			if (u < cumSum) return i;
		}
		return n - 1; // fallback
	}

	public void Dispose()
	{
		_session?.Dispose();
	}
}
