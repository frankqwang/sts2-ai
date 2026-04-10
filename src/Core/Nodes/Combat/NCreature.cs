using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Animation;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Powers;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Monsters;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Multiplayer;
using MegaCrit.Sts2.Core.Nodes.Orbs;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.ScreenContext;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NCreature : Control
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("combat/creature");

	private NCreatureStateDisplay _stateDisplay;

	private Tween? _intentFadeTween;

	private Tween? _shakeTween;

	private CreatureAnimator? _spineAnimator;

	private bool _isRemotePlayerOrPet;

	private float _tempScale = 1f;

	private Tween? _scaleTween;

	private bool _isInMultiselect;

	private NSelectionReticle _selectionReticle;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public Task? DeathAnimationTask { get; set; }

	public CancellationTokenSource DeathAnimCancelToken { get; } = new CancellationTokenSource();

	public Control Hitbox { get; private set; }

	public NOrbManager? OrbManager { get; private set; }

	public bool IsInteractable { get; private set; } = true;

	public Creature Entity { get; private set; }

	public Vector2 VfxSpawnPosition => Visuals.VfxSpawnPosition.GlobalPosition;

	public NCreatureVisuals Visuals { get; private set; }

	public Node2D Body => Visuals.Body;

	public Control IntentContainer { get; private set; }

	public bool IsPlayingDeathAnimation => DeathAnimationTask != null;

	public bool HasSpineAnimation => Visuals.HasSpineAnimation;

	public MegaSprite? SpineController => Visuals.SpineBody;

	public bool IsFocused { get; private set; }

	public NMultiplayerPlayerIntentHandler? PlayerIntentHandler { get; private set; }

	public T? GetSpecialNode<T>(string name) where T : Node
	{
		return Visuals.GetNode<T>(name);
	}

	public static NCreature? Create(Creature entity)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NCreature>(PackedScene.GenEditState.Disabled);
		nCreature.Entity = entity;
		nCreature.Visuals = entity.CreateVisuals();
		return nCreature;
	}

	public override void _Ready()
	{
		_stateDisplay = GetNode<NCreatureStateDisplay>("%HealthBar");
		IntentContainer = GetNode<Control>("%Intents");
		Hitbox = GetNode<Control>("%Hitbox");
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		Hitbox.Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Hitbox.Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		Hitbox.Connect(Control.SignalName.MouseEntered, Callable.From(OnFocus));
		Hitbox.Connect(Control.SignalName.MouseExited, Callable.From(OnUnfocus));
		if (Entity.IsPlayer)
		{
			OrbManager = NOrbManager.Create(this, LocalContext.IsMe(Entity));
			this.AddChildSafely(OrbManager);
			UpdateNavigation();
		}
		if (Entity.IsPlayer)
		{
			CombatState? combatState = Entity.CombatState;
			if (combatState != null && combatState.RunState.Players.Count > 1)
			{
				PlayerIntentHandler = NMultiplayerPlayerIntentHandler.Create(Entity.Player);
				if (PlayerIntentHandler != null)
				{
					IntentContainer.AddChildSafely(PlayerIntentHandler);
					IntentContainer.Modulate = Colors.White;
				}
			}
		}
		this.AddChildSafely(Visuals);
		MoveChild(Visuals, 0);
		Visuals.Position = Vector2.Zero;
		_stateDisplay.SetCreature(Entity);
		bool flag = Entity.PetOwner != null && !LocalContext.IsMe(Entity.PetOwner);
		bool flag2 = Entity.IsPlayer && !LocalContext.IsMe(Entity);
		_isRemotePlayerOrPet = flag2 || flag;
		if (_isRemotePlayerOrPet)
		{
			_stateDisplay.HideImmediately();
		}
		else
		{
			bool flag3 = NCombatRoom.Instance != null && Time.GetTicksMsec() - NCombatRoom.Instance.CreatedMsec < 1000;
			_stateDisplay.AnimateIn(flag3 ? HealthBarAnimMode.SpawnedAtCombatStart : HealthBarAnimMode.SpawnedDuringCombat);
		}
		if (HasSpineAnimation)
		{
			if (Entity.Player != null)
			{
				_spineAnimator = Entity.Player.Character.GenerateAnimator(SpineController);
			}
			else
			{
				_spineAnimator = Entity.Monster.GenerateAnimator(SpineController);
				Visuals.SetUpSkin(Entity.Monster);
			}
			ConnectSpineAnimatorSignals();
			if (Entity.IsDead)
			{
				SetAnimationTrigger("Dead");
				MegaTrackEntry current = Visuals.SpineBody.GetAnimationState().GetCurrent(0);
				current.SetTrackTime(current.GetAnimationEnd());
			}
		}
		SetOrbManagerPosition();
		if (Entity.Monster != null)
		{
			ToggleIsInteractable(Entity.Monster.IsHealthBarVisible);
		}
		UpdateBounds(Visuals);
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		CombatManager.Instance.CombatEnded += OnCombatEnded;
		Entity.PowerApplied += OnPowerApplied;
		Entity.PowerRemoved += OnPowerRemoved;
		Entity.PowerIncreased += OnPowerIncreased;
		foreach (PowerModel power in Entity.Powers)
		{
			SubscribeToPower(power);
		}
		ConnectSpineAnimatorSignals();
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		DeathAnimCancelToken.Cancel();
		CombatManager.Instance.CombatEnded -= OnCombatEnded;
		Entity.PowerApplied -= OnPowerApplied;
		Entity.PowerRemoved -= OnPowerRemoved;
		Entity.PowerIncreased -= OnPowerIncreased;
		foreach (PowerModel power in Entity.Powers)
		{
			UnsubscribeFromPower(power);
		}
		if (_spineAnimator != null)
		{
			_spineAnimator.BoundsUpdated -= UpdateBounds;
		}
		CombatManager.Instance.StateTracker.CombatStateChanged -= ShowCreatureHoverTips;
	}

	private void ConnectSpineAnimatorSignals()
	{
		if (_spineAnimator != null)
		{
			_spineAnimator.BoundsUpdated -= UpdateBounds;
			_spineAnimator.BoundsUpdated += UpdateBounds;
		}
	}

	private void UpdateBounds(string boundsNodeName)
	{
		UpdateBounds(Visuals.GetNode<Control>(boundsNodeName));
	}

	private void UpdateBounds(Node boundsContainer)
	{
		Control node = boundsContainer.GetNode<Control>("Bounds");
		Vector2 size = node.Size * Visuals.Scale / _tempScale;
		Vector2 vector = (node.GlobalPosition - base.GlobalPosition) / _tempScale;
		Hitbox.Size = size;
		Hitbox.GlobalPosition = base.GlobalPosition + vector;
		_selectionReticle.Size = size;
		_selectionReticle.GlobalPosition = base.GlobalPosition + vector;
		_selectionReticle.PivotOffset = _selectionReticle.Size / 2f;
		IntentContainer.Position = boundsContainer.GetNode<Marker2D>("IntentPos").Position - IntentContainer.Size / 2f;
		_stateDisplay.SetCreatureBounds(Hitbox);
	}

	public void UpdateNavigation()
	{
		if (OrbManager != null)
		{
			Hitbox.FocusNeighborTop = OrbManager.DefaultFocusOwner.GetPath();
		}
	}

	public Task UpdateIntent(IEnumerable<Creature> targets)
	{
		if (Entity.Monster == null)
		{
			throw new InvalidOperationException("Only valid on monsters.");
		}
		IReadOnlyList<AbstractIntent> intents = Entity.Monster.NextMove.Intents;
		int i;
		for (i = 0; i < intents.Count && i < IntentContainer.GetChildCount(); i++)
		{
			NIntent child = IntentContainer.GetChild<NIntent>(i);
			child.SetFrozen(isFrozen: false);
			child.UpdateIntent(intents[i], targets, Entity);
		}
		float num = (float)GetHashCode() / 100f;
		for (; i < intents.Count; i++)
		{
			NIntent nIntent = NIntent.Create(num + (float)i * 0.3f);
			IntentContainer.AddChildSafely(nIntent);
			nIntent.UpdateIntent(intents[i], targets, Entity);
		}
		List<Node> list = IntentContainer.GetChildren().TakeLast(IntentContainer.GetChildCount() - i).ToList();
		foreach (Node item in list)
		{
			IntentContainer.RemoveChildSafely(item);
			item.QueueFreeSafely();
		}
		return Task.CompletedTask;
	}

	public async Task PerformIntent()
	{
		foreach (NIntent item in IntentContainer.GetChildren().OfType<NIntent>())
		{
			item.PlayPerform();
			item.SetFrozen(isFrozen: true);
		}
		if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			IntentContainer.Modulate = new Color(IntentContainer.Modulate.R, IntentContainer.Modulate.G, IntentContainer.Modulate.B, 0f);
			return;
		}
		AnimHideIntent(0.4f);
		await Cmd.CustomScaledWait(0.25f, 0.4f);
	}

	public async Task RefreshIntents()
	{
		await UpdateIntent(Entity.CombatState.Players.Select((Player p) => p.Creature));
		await RevealIntents();
	}

	private Task RevealIntents()
	{
		IntentContainer.Modulate = Colors.Transparent;
		_intentFadeTween?.Kill();
		_intentFadeTween = CreateTween().SetParallel();
		_intentFadeTween.TweenProperty(IntentContainer, "modulate:a", 1f, 1.0).SetDelay(Rng.Chaotic.NextFloat(0f, 0.3f));
		return Task.CompletedTask;
	}

	private void OnFocus()
	{
		if (IsFocused)
		{
			return;
		}
		IsFocused = true;
		if (_isRemotePlayerOrPet)
		{
			_stateDisplay.AnimateIn(HealthBarAnimMode.FromHidden);
			_stateDisplay.ZIndex = 1;
			Player me = LocalContext.GetMe(Entity.CombatState);
			NCombatRoom.Instance?.GetCreatureNode(me?.Creature)?.SetRemotePlayerFocused(remotePlayerFocused: true);
		}
		else
		{
			_stateDisplay.ShowNameplate();
		}
		NRun.Instance.GlobalUi.MultiplayerPlayerContainer.HighlightPlayer(Entity.Player);
		if (NTargetManager.Instance.IsInSelection)
		{
			NTargetManager.Instance.OnNodeHovered(this);
			return;
		}
		if (NControllerManager.Instance.IsUsingController)
		{
			ShowSingleSelectReticle();
		}
		ShowHoverTips(Entity.HoverTips);
		CombatManager.Instance.StateTracker.CombatStateChanged += ShowCreatureHoverTips;
	}

	private void OnUnfocus()
	{
		IsFocused = false;
		HideSingleSelectReticle();
		if (_isRemotePlayerOrPet)
		{
			_stateDisplay.AnimateOut();
			Player me = LocalContext.GetMe(Entity.CombatState);
			NCombatRoom.Instance?.GetCreatureNode(me?.Creature)?.SetRemotePlayerFocused(remotePlayerFocused: false);
		}
		else
		{
			_stateDisplay.HideNameplate();
		}
		NRun.Instance.GlobalUi.MultiplayerPlayerContainer.UnhighlightPlayer(Entity.Player);
		NTargetManager.Instance.OnNodeUnhovered(this);
		CombatManager.Instance.StateTracker.CombatStateChanged -= ShowCreatureHoverTips;
		HideHoverTips();
	}

	public void OnTargetingStarted()
	{
		if (IsFocused)
		{
			NTargetManager.Instance.OnNodeHovered(this);
			CombatManager.Instance.StateTracker.CombatStateChanged -= ShowCreatureHoverTips;
			HideHoverTips();
		}
	}

	private void ShowCreatureHoverTips(CombatState _)
	{
		if (Entity.CombatState != null)
		{
			ShowHoverTips(Entity.HoverTips);
		}
	}

	public void ShowHoverTips(IEnumerable<IHoverTip> hoverTips)
	{
		if (!NCombatRoom.Instance.Ui.Hand.InCardPlay)
		{
			HideHoverTips();
			NHoverTipSet.CreateAndShow(Hitbox, hoverTips, HoverTip.GetHoverTipAlignment(this, 0.5f));
		}
	}

	public void SetRemotePlayerFocused(bool remotePlayerFocused)
	{
		if (!LocalContext.IsMe(Entity))
		{
			throw new InvalidOperationException("This should only be called on the local player's creature node!");
		}
		if (remotePlayerFocused)
		{
			_stateDisplay.AnimateOut();
		}
		else if (Entity.IsAlive)
		{
			_stateDisplay.AnimateIn(HealthBarAnimMode.FromHidden);
		}
	}

	public void HideHoverTips()
	{
		NHoverTipSet.Remove(Hitbox);
	}

	private void SubscribeToPower(PowerModel power)
	{
		power.Flashed += OnPowerFlashed;
	}

	private void UnsubscribeFromPower(PowerModel power)
	{
		power.Flashed -= OnPowerFlashed;
	}

	private void OnPowerApplied(PowerModel power)
	{
		SubscribeToPower(power);
	}

	private void OnPowerIncreased(PowerModel power, int amount, bool silent)
	{
		if (silent || !CombatManager.Instance.IsInProgress)
		{
			return;
		}
		NPowerAppliedVfx vfx = NPowerAppliedVfx.Create(power, amount);
		if (vfx != null)
		{
			Callable.From(delegate
			{
				NCombatRoom.Instance?.CombatVfxContainer.AddChildSafely(vfx);
			}).CallDeferred();
		}
		if (power.ShouldPlayVfx)
		{
			SfxCmd.Play((power.GetTypeForAmount(amount) == PowerType.Buff) ? "event:/sfx/buff" : "event:/sfx/debuff");
		}
		if (power.GetTypeForAmount(power.Amount) == PowerType.Debuff)
		{
			AnimShake();
		}
	}

	private void OnPowerRemoved(PowerModel power)
	{
		NPowerRemovedVfx vfx = NPowerRemovedVfx.Create(power);
		if (vfx != null)
		{
			Callable.From(delegate
			{
				NCombatRoom.Instance?.CombatVfxContainer.AddChildSafely(vfx);
			}).CallDeferred();
		}
		UnsubscribeFromPower(power);
	}

	private void OnPowerFlashed(PowerModel power)
	{
		NPowerFlashVfx vfx = NPowerFlashVfx.Create(power);
		if (vfx != null)
		{
			Callable.From(delegate
			{
				NCombatRoom.Instance?.CombatVfxContainer.AddChildSafely(vfx);
			}).CallDeferred();
		}
	}

	private void OnCombatEnded(CombatRoom _)
	{
		AnimHideIntent();
		OrbManager?.ClearOrbs();
	}

	public void SetAnimationTrigger(string trigger)
	{
		_spineAnimator?.SetTrigger(trigger);
	}

	public float GetCurrentAnimationLength()
	{
		return SpineController.GetAnimationState().GetCurrent(0).GetAnimation()
			.GetDuration();
	}

	public float GetCurrentAnimationTimeRemaining()
	{
		MegaTrackEntry current = SpineController.GetAnimationState().GetCurrent(0);
		return current.GetTrackComplete() - current.GetTrackTime();
	}

	public void ToggleIsInteractable(bool on)
	{
		IsInteractable = on;
		_stateDisplay.Visible = !NCombatUi.IsDebugHidingHpBar && on;
		Hitbox.MouseFilter = (MouseFilterEnum)(on ? 0 : 2);
	}

	public Tween AnimDisableUi()
	{
		Tween tween = CreateTween();
		if (!IsNodeReady())
		{
			tween.TweenInterval(0.0);
			return tween;
		}
		tween.TweenProperty(_stateDisplay, "modulate:a", 0f, 0.5).SetDelay(0.5).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Expo);
		return tween;
	}

	public Tween AnimEnableUi()
	{
		Tween tween = CreateTween();
		tween.TweenProperty(_stateDisplay, "modulate:a", 1f, 0.5).SetDelay(0.5).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Expo);
		return tween;
	}

	public float StartDeathAnim(bool shouldRemove)
	{
		if (Hitbox.HasFocus())
		{
			ActiveScreenContext.Instance.FocusOnDefaultControl();
		}
		Hitbox.FocusMode = FocusModeEnum.None;
		foreach (NIntent item in IntentContainer.GetChildren().OfType<NIntent>())
		{
			item.SetFrozen(isFrozen: true);
		}
		Task deathAnimationTask = DeathAnimationTask;
		if (deathAnimationTask != null && !deathAnimationTask.IsCompleted)
		{
			return 0f;
		}
		float a = 0f;
		if (_spineAnimator != null)
		{
			MonsterModel? monster = Entity.Monster;
			if (monster != null && monster.HasDeathSfx)
			{
				SfxCmd.PlayDeath(Entity.Monster);
			}
			if (Entity.Player != null)
			{
				SfxCmd.PlayDeath(Entity.Player);
			}
			SetAnimationTrigger("Dead");
			a = GetCurrentAnimationLength();
		}
		DeathAnimationTask = AnimDie(shouldRemove, DeathAnimCancelToken.Token);
		TaskHelper.RunSafely(DeathAnimationTask);
		MonsterModel monster2 = Entity.Monster;
		if (monster2 != null && monster2.HasDeathAnimLengthOverride)
		{
			return Entity.Monster.DeathAnimLengthOverride;
		}
		return Mathf.Min(a, 30f);
	}

	public void StartReviveAnim()
	{
		CreatureAnimator? spineAnimator = _spineAnimator;
		if (spineAnimator != null && spineAnimator.HasTrigger("Revive"))
		{
			SetAnimationTrigger("Revive");
		}
		else if (Entity.IsPlayer)
		{
			AnimTempRevive();
		}
		if (!_isRemotePlayerOrPet)
		{
			AnimEnableUi();
		}
		Hitbox.MouseFilter = MouseFilterEnum.Stop;
	}

	private void AnimTempRevive()
	{
		Tween tween = CreateTween();
		tween.TweenProperty(Visuals, "modulate:a", 0f, 0.20000000298023224);
		tween.TweenCallback(Callable.From(ImmediatelySetIdle));
		tween.TweenProperty(Visuals, "modulate:a", 1f, 0.20000000298023224);
	}

	private void ImmediatelySetIdle()
	{
		_spineAnimator?.SetTrigger("Idle");
		MegaTrackEntry current = Visuals.SpineBody.GetAnimationState().GetCurrent(0);
		current.SetMixDuration(0f);
		current.SetTrackTime(current.GetAnimationEnd());
	}

	private async Task AnimDie(bool shouldRemove, CancellationToken cancelToken)
	{
		Tween disableUiTween = AnimDisableUi();
		Hitbox.MouseFilter = MouseFilterEnum.Ignore;
		if (!RunManager.Instance.IsSinglePlayerOrFakeMultiplayer)
		{
			OrbManager?.ClearOrbs();
		}
		if (shouldRemove)
		{
			AnimHideIntent();
		}
		if (_spineAnimator != null)
		{
			float seconds = Math.Min(GetCurrentAnimationTimeRemaining() + 0.5f, 20f);
			await Cmd.Wait(seconds, cancelToken, ignoreCombatEnd: true);
			if (cancelToken.IsCancellationRequested)
			{
				return;
			}
		}
		else
		{
			MonsterModel monster = Entity.Monster;
			if (monster != null && monster.HasDeathAnimLengthOverride)
			{
				await Cmd.Wait(Entity.Monster.DeathAnimLengthOverride, cancelToken, ignoreCombatEnd: true);
			}
		}
		if (shouldRemove)
		{
			Task fadeVfx = null;
			MonsterModel monster = Entity.Monster;
			if (monster != null && monster.ShouldFadeAfterDeath && Body.IsVisibleInTree())
			{
				NMonsterDeathVfx nMonsterDeathVfx = NMonsterDeathVfx.Create(this, cancelToken);
				Node parent = GetParent();
				parent.AddChildSafely(nMonsterDeathVfx);
				parent.MoveChild(nMonsterDeathVfx, GetIndex());
				fadeVfx = nMonsterDeathVfx?.PlayVfx();
			}
			if (SaveManager.Instance.PrefsSave.FastMode != FastModeType.Instant)
			{
				if (disableUiTween.IsValid() && disableUiTween.IsRunning())
				{
					await ToSignal(disableUiTween, Tween.SignalName.Finished);
				}
				foreach (IDeathDelayer item in this.GetChildrenRecursive<IDeathDelayer>())
				{
					await item.GetDelayTask();
				}
			}
			if (fadeVfx != null)
			{
				await fadeVfx;
			}
			this.QueueFreeSafely();
		}
		if (Entity.Monster is Osty)
		{
			OstyScaleToSize(0f, 0.75f);
		}
	}

	public void AnimHideIntent(float delay = 0f)
	{
		_intentFadeTween?.Kill();
		_intentFadeTween = CreateTween().SetParallel();
		PropertyTweener propertyTweener = _intentFadeTween.TweenProperty(IntentContainer, "modulate:a", 0f, 0.5);
		if (delay > 0f)
		{
			propertyTweener.SetDelay(delay);
		}
	}

	public void SetScaleAndHue(float scale, float hue)
	{
		Visuals.SetScaleAndHue(scale, hue);
		UpdateBounds(Visuals);
	}

	public void ScaleTo(float size, float duration)
	{
		if (!Entity.IsMonster || Entity.Monster.CanChangeScale)
		{
			_tempScale = size;
			_scaleTween?.Kill();
			_scaleTween = CreateTween();
			_scaleTween.TweenMethod(Callable.From<Vector2>(DoScaleTween), Visuals.Scale, Vector2.One * _tempScale * Visuals.DefaultScale, duration).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
		}
	}

	public void SetDefaultScaleTo(float size, float duration)
	{
		if (!Entity.IsMonster || Entity.Monster.CanChangeScale)
		{
			Visuals.DefaultScale = size;
			ScaleTo(_tempScale, duration);
		}
	}

	public void OstyScaleToSize(float ostyHealth, float duration)
	{
		float num = Mathf.Lerp(Osty.ScaleRange.X, Osty.ScaleRange.Y, Mathf.Clamp(ostyHealth / 150f, 0f, 1f));
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(Entity.PetOwner.Creature);
		_scaleTween = CreateTween();
		_scaleTween.TweenProperty(Visuals, "scale", Vector2.One * num * Visuals.DefaultScale, duration).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
		if (LocalContext.IsMe(Entity.PetOwner))
		{
			_scaleTween.Parallel().TweenProperty(this, "position", nCreature.Position + GetOstyOffsetFromPlayer(Entity), duration);
		}
		_scaleTween.TweenCallback(Callable.From(delegate
		{
			UpdateBounds(Visuals);
		}));
	}

	public static Vector2 GetOstyOffsetFromPlayer(Creature osty)
	{
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(osty.PetOwner.Creature);
		return Vector2.Right * nCreature.Hitbox.Size.X * 0.5f + Osty.MinOffset.Lerp(Osty.MaxOffset, Mathf.Clamp((float)osty.MaxHp / 150f, 0f, 1f));
	}

	public void AnimShake()
	{
		if ((_shakeTween == null || !_shakeTween.IsRunning()) && !Visuals.IsPlayingHurtAnimation())
		{
			Visuals.Position = Vector2.Zero;
			_shakeTween = CreateTween();
			_shakeTween.TweenMethod(Callable.From(delegate(float t)
			{
				Visuals.Position = Vector2.Right * 10f * Mathf.Sin(t * 4f) * Mathf.Sin(t / 2f);
			}), 0f, (float)Math.PI * 2f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		}
	}

	private void DoScaleTween(Vector2 scale)
	{
		Visuals.Scale = scale;
		SetOrbManagerPosition();
	}

	private void SetOrbManagerPosition()
	{
		if (OrbManager != null)
		{
			OrbManager.Scale = ((Visuals.Scale.X > 1f) ? Vector2.One : Visuals.Scale.Lerp(Vector2.One, 0.5f));
			OrbManager.Position = Visuals.OrbPosition.Position * Mathf.Min(Visuals.Scale.X, 1.25f);
			if (!OrbManager.IsLocal)
			{
				OrbManager.Position += Vector2.Up * 50f;
			}
		}
	}

	public Vector2 GetTopOfHitbox()
	{
		return Hitbox.GlobalPosition + new Vector2(Hitbox.Size.X * 0.5f, 0f);
	}

	public Vector2 GetBottomOfHitbox()
	{
		return Hitbox.GlobalPosition + new Vector2(Hitbox.Size.X * 0.5f, Hitbox.Size.Y);
	}

	public void TrackBlockStatus(Creature creature)
	{
		_stateDisplay.TrackBlockStatus(creature);
	}

	public void ShowMultiselectReticle()
	{
		_isInMultiselect = true;
		ShowSingleSelectReticle();
	}

	public void HideMultiselectReticle()
	{
		_isInMultiselect = false;
		HideSingleSelectReticle();
	}

	public void ShowSingleSelectReticle()
	{
		_selectionReticle.OnSelect();
	}

	public void HideSingleSelectReticle()
	{
		if (!_isInMultiselect)
		{
			_selectionReticle.OnDeselect();
		}
	}
}
