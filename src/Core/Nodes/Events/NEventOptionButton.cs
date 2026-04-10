using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Events;

public partial class NEventOptionButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _h = new StringName("h");

	private static readonly string _voteIconPath = SceneHelper.GetScenePath("ui/multiplayer_vote_icon");

	private MegaRichTextLabel _label;

	private NinePatchRect _image;

	private NinePatchRect _outline;

	private NinePatchRect _killGlow;

	private NinePatchRect _confirmFlash;

	private ShaderMaterial? _hsv;

	private NMultiplayerVoteContainer _playerVoteContainer;

	private Tween? _animInTween;

	private Tween? _flashTween;

	private Tween? _killGlowTween;

	private Tween? _tween;

	private static readonly Vector2 _hoverScale = Vector2.One * 1.01f;

	private static readonly Vector2 _pressScale = Vector2.One * 0.99f;

	private const float _defaultV = 0.9f;

	private const float _hoverV = 1.2f;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	private Color _buttonColor;

	private NThoughtBubbleVfx? _deathPreventionVfx;

	private CancellationTokenSource? _deathPreventionCancellation;

	private Vector2 _deathPreventionVfxPosition;

	public EventModel Event { get; private set; }

	public EventOption Option { get; private set; }

	private int Index { get; set; }

	public NMultiplayerVoteContainer VoteContainer => _playerVoteContainer;

	private static string ScenePath => SceneHelper.GetScenePath("events/event_option_button");

	private static string AncientScenePath => SceneHelper.GetScenePath("events/ancient_event_option_button");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[3] { ScenePath, AncientScenePath, _voteIconPath });

	public static NEventOptionButton Create(EventModel eventModel, EventOption option, int index)
	{
		NEventOptionButton nEventOptionButton = ((eventModel is AncientEventModel) ? PreloadManager.Cache.GetScene(AncientScenePath).Instantiate<NEventOptionButton>(PackedScene.GenEditState.Disabled) : PreloadManager.Cache.GetScene(ScenePath).Instantiate<NEventOptionButton>(PackedScene.GenEditState.Disabled));
		nEventOptionButton.Event = eventModel;
		nEventOptionButton.Option = option;
		nEventOptionButton.Index = index;
		nEventOptionButton._buttonColor = eventModel.ButtonColor;
		return nEventOptionButton;
	}

	public override void _Ready()
	{
		ConnectSignals();
		Event.DynamicVars.AddTo(Option.Description);
		Event.DynamicVars.AddTo(Option.Title);
		string formattedText = Option.Title.GetFormattedText();
		string formattedText2 = Option.Description.GetFormattedText();
		_label = GetNode<MegaRichTextLabel>("%Text");
		if (string.IsNullOrEmpty(formattedText))
		{
			_label.Text = formattedText2;
		}
		else
		{
			string value = (Option.IsLocked ? "red" : "gold");
			_label.Text = $"[{value}][b]{formattedText}[/b][/{value}]\n{formattedText2}";
		}
		_image = GetNode<NinePatchRect>("Image");
		_image.Modulate = _buttonColor;
		_hsv = (ShaderMaterial)_image.Material;
		_outline = GetNode<NinePatchRect>("Outline");
		_killGlow = GetNode<NinePatchRect>("RedFlash");
		_confirmFlash = GetNode<NinePatchRect>("BlueFlash");
		_playerVoteContainer = GetNode<NMultiplayerVoteContainer>("PlayerVoteContainer");
		_playerVoteContainer.Initialize(ShouldDisplayPlayerVote, Event.Owner.RunState.Players);
		if (Event is AncientEventModel && Option.Relic != null)
		{
			TextureRect node = GetNode<TextureRect>("%RelicIcon");
			node.SetTexture(Option.Relic.Icon);
			node.GetNode<TextureRect>("%Outline").SetTexture(Option.Relic.IconOutline);
			node.Visible = true;
		}
		if (Option.IsLocked)
		{
			SetVisuallyLocked();
		}
		else if (WillKillPlayer())
		{
			ShowPersistentKillGlow();
		}
	}

	private void SetVisuallyLocked()
	{
		_label.Modulate = new Color(_label.Modulate.R, _label.Modulate.G, _label.Modulate.B, 0.7f);
		_hsv?.SetShaderParameter(_h, 0.48f);
		_hsv?.SetShaderParameter(_s, 0.2f);
		_hsv?.SetShaderParameter(_v, 0.65f);
	}

	public void AnimateIn()
	{
		if (GodotObject.IsInstanceValid(this))
		{
			if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
			{
				EnableButton();
				return;
			}
			base.Modulate = StsColors.transparentWhite;
			bool flag = SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast;
			_animInTween = CreateTween().SetParallel();
			_animInTween.TweenInterval(flag ? 0.25 : 0.5);
			_animInTween.Chain();
			_animInTween.TweenInterval((flag ? 0.1 : 0.2) * (double)Index);
			_animInTween.Chain();
			_animInTween.TweenProperty(this, "position", base.Position, flag ? 0.25 : 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
				.From(base.Position + new Vector2(-60f, 0f));
			_animInTween.TweenProperty(this, "modulate", Colors.White, flag ? 0.25 : 0.5);
			_animInTween.Finished += EnableButton;
		}
	}

	public void EnableButton()
	{
		base.MouseFilter = MouseFilterEnum.Stop;
	}

	protected override void OnRelease()
	{
		if (Option.IsLocked)
		{
			return;
		}
		if (WillKillPlayer())
		{
			if (Event.Owner.RunState.Players.Count > 1)
			{
				if (_deathPreventionVfx == null)
				{
					LocString eventDeathPreventionLine = Event.Owner.Character.EventDeathPreventionLine;
					_deathPreventionVfx = NThoughtBubbleVfx.Create(eventDeathPreventionLine.GetFormattedText(), DialogueSide.Right, null);
					NEventRoom.Instance?.VfxContainer?.AddChildSafely(_deathPreventionVfx);
					_deathPreventionVfxPosition = base.GlobalPosition + new Vector2(-50f, -15f);
					_deathPreventionVfx.GlobalPosition = _deathPreventionVfxPosition;
				}
				else
				{
					_deathPreventionCancellation?.Cancel();
				}
				TaskHelper.RunSafely(RumbleDeathVfx());
				TaskHelper.RunSafely(ExpireDeathPreventionVfx());
				return;
			}
			ShowPersistentKillGlow();
		}
		NEventRoom.Instance.OptionButtonClicked(Option, Index);
	}

	private async Task RumbleDeathVfx()
	{
		ScreenRumbleInstance rumble = new ScreenRumbleInstance(20f, 0.30000001192092896, 10f, RumbleStyle.Rumble);
		while (!rumble.IsDone)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			Vector2 vector = rumble.Update(GetProcessDeltaTime());
			_deathPreventionVfx.GlobalPosition = _deathPreventionVfxPosition + vector;
		}
	}

	private async Task ExpireDeathPreventionVfx()
	{
		_deathPreventionCancellation = new CancellationTokenSource();
		await Cmd.Wait(2.5f, _deathPreventionCancellation.Token, ignoreCombatEnd: true);
		if (!_deathPreventionCancellation.IsCancellationRequested)
		{
			_deathPreventionVfx?.GoAway();
			_deathPreventionVfx = null;
		}
	}

	protected override void OnPress()
	{
		if (!Option.IsLocked)
		{
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(this, "scale", _pressScale, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_tween.TweenMethod(Callable.From<float>(UpdateShaderParam), 1.2f, 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			base.OnPress();
		}
	}

	protected override void OnFocus()
	{
		if (!Option.IsLocked)
		{
			base.OnFocus();
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(this, "scale", _hoverScale, 0.05);
			_tween.TweenProperty(_outline, "modulate", StsColors.blueGlow, 0.05);
			_hsv?.SetShaderParameter(_v, 1.2f);
			NHoverTipSet.CreateAndShow(this, Option.HoverTips, (Event.LayoutType != EventLayoutType.Combat) ? HoverTipAlignment.Left : HoverTipAlignment.Right);
			if (WillKillPlayer())
			{
				PulseKillGlow();
			}
		}
	}

	protected override void OnUnfocus()
	{
		if (!Option.IsLocked)
		{
			base.OnUnfocus();
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_tween.TweenMethod(Callable.From<float>(UpdateShaderParam), 1.2f, 0.9f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_tween.TweenProperty(_outline, "modulate:a", 0f, 0.3);
			NHoverTipSet.Remove(this);
			if (WillKillPlayer())
			{
				ShowPersistentKillGlow();
			}
		}
	}

	public async Task FlashConfirmation()
	{
		_flashTween?.Kill();
		NinePatchRect confirmFlash = _confirmFlash;
		Color modulate = _confirmFlash.Modulate;
		modulate.A = 0f;
		confirmFlash.Modulate = modulate;
		_flashTween = CreateTween();
		_flashTween.TweenProperty(_confirmFlash, "modulate:a", 0.8f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_flashTween.Parallel().TweenProperty(_confirmFlash, "scale", new Vector2(1.03f, 1.1f), 0.25).SetEase(Tween.EaseType.Out)
			.SetTrans(Tween.TransitionType.Expo);
		_flashTween.TweenProperty(_confirmFlash, "scale", Vector2.One, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_flashTween.TweenProperty(_confirmFlash, "modulate:a", 0f, 0.3);
		await ToSignal(_flashTween, Tween.SignalName.Finished);
		await Cmd.Wait(0.5f);
	}

	public void GrayOut()
	{
		_flashTween?.Kill();
		_flashTween = CreateTween();
		Tween? flashTween = _flashTween;
		NodePath property = "modulate";
		Color lightGray = StsColors.lightGray;
		lightGray.A = 0.5f;
		flashTween.TweenProperty(this, property, lightGray, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	private void UpdateShaderParam(float newV)
	{
		_hsv?.SetShaderParameter(_v, newV);
	}

	private bool ShouldDisplayPlayerVote(Player player)
	{
		return RunManager.Instance.EventSynchronizer.GetPlayerVote(player) == Index;
	}

	public void RefreshVotes()
	{
		_playerVoteContainer.RefreshPlayerVotes();
	}

	public override void _ExitTree()
	{
		_cancelToken.Cancel();
		if (_animInTween != null)
		{
			_animInTween.Finished -= EnableButton;
		}
	}

	private bool WillKillPlayer()
	{
		if (Event.Owner != null)
		{
			return Option.WillKillPlayer?.Invoke(Event.Owner) ?? false;
		}
		return false;
	}

	private void ShowPersistentKillGlow()
	{
		_killGlowTween?.Kill();
		NinePatchRect killGlow = _killGlow;
		Color modulate = _killGlow.Modulate;
		modulate.A = 0.5f;
		killGlow.Modulate = modulate;
	}

	private void PulseKillGlow()
	{
		_killGlowTween?.Kill();
		NinePatchRect killGlow = _killGlow;
		Color modulate = _killGlow.Modulate;
		modulate.A = 0.25f;
		killGlow.Modulate = modulate;
		_killGlowTween = CreateTween();
		_killGlowTween.TweenProperty(_killGlow, "modulate:a", 1f, 0.2).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_killGlowTween.TweenProperty(_killGlow, "modulate:a", 0.25f, 0.8);
		_killGlowTween.SetLoops();
	}
}
