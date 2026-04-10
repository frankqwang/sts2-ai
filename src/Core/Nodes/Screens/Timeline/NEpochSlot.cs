using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Timeline;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Timeline;

public partial class NEpochSlot : NButton
{
	private static readonly StringName _lod = new StringName("lod");

	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private const string _unlockIconPath = "res://images/packed/unlock_icon.png";

	private const string _scenePath = "res://scenes/timeline_screen/epoch_slot.tscn";

	public static readonly IEnumerable<string> assetPaths = new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { "res://scenes/timeline_screen/epoch_slot.tscn", "res://images/packed/unlock_icon.png" });

	private TextureRect _slotImage;

	private TextureRect _portrait;

	private TextureRect _chains;

	private ShaderMaterial _hsv;

	private TextureRect _blurPortrait;

	private TextureRect _outline;

	private Control _blur;

	private ShaderMaterial _blurShader;

	private NSelectionReticle _selectionReticle;

	private NEpochOffscreenVfx? _offscreenVfx;

	private Control? _highlightVfx;

	private SubViewportContainer _subViewportContainer;

	private SubViewport _subViewport;

	private bool _isGlowPulsing;

	public EpochModel model;

	private bool _isComplete;

	private bool _isHovered;

	private EpochEra _era;

	public int eraPosition;

	private Tween? _glowTween;

	private Tween? _spawnTween;

	private Tween? _hoverTween;

	private static readonly Color _highlightSlotColor = StsColors.purple;

	private static readonly Color _defaultSlotOutlineColor = new Color("70a0ff18");

	private IHoverTip? _hoverTip;

	protected override string ClickedSfx => "event:/sfx/ui/timeline/ui_timeline_click";

	protected override string HoveredSfx
	{
		get
		{
			if (State != EpochSlotState.NotObtained)
			{
				return "event:/sfx/ui/timeline/ui_timeline_hover";
			}
			return "event:/sfx/ui/timeline/ui_timeline_hover_locked";
		}
	}

	public EpochSlotState State { get; private set; }

	public bool HasSpawned { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		_slotImage = GetNode<TextureRect>("%SlotImage");
		_portrait = GetNode<TextureRect>("%Portrait");
		_chains = GetNode<TextureRect>("%Chains");
		_hsv = (ShaderMaterial)_portrait.GetMaterial();
		_blurPortrait = GetNode<TextureRect>("%BlurPortrait");
		_outline = GetNode<TextureRect>("%Outline");
		_subViewportContainer = GetNode<SubViewportContainer>("%SubViewportContainer");
		_subViewport = GetNode<SubViewport>("%SubViewport");
		_blurShader = (ShaderMaterial)GetNode<Control>("%Blur").GetMaterial();
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		if (!NGame.IsReleaseGame())
		{
			MegaLabel node = GetNode<MegaLabel>("%DebugLabel");
			node.Text = model.GetType().Name;
			node.Visible = true;
		}
		SetState(State);
	}

	public static NEpochSlot Create(EpochSlotData data)
	{
		NEpochSlot nEpochSlot = PreloadManager.Cache.GetScene("res://scenes/timeline_screen/epoch_slot.tscn").Instantiate<NEpochSlot>(PackedScene.GenEditState.Disabled);
		nEpochSlot._era = data.Era;
		nEpochSlot.State = data.State;
		nEpochSlot.eraPosition = data.EraPosition;
		nEpochSlot.model = data.Model;
		return nEpochSlot;
	}

	protected override void OnRelease()
	{
		if (!NGame.IsReleaseGame() && State == EpochSlotState.NotObtained && Input.IsKeyPressed(Key.Ctrl))
		{
			NHoverTipSet.Remove(this);
			State = EpochSlotState.Obtained;
		}
		base.OnRelease();
		if (State == EpochSlotState.Obtained)
		{
			NTimelineScreen.Instance.DisableInput();
			GetViewport().GuiReleaseFocus();
			RevealEpoch();
			SetState(EpochSlotState.Complete);
		}
		else if (State == EpochSlotState.Complete)
		{
			NTimelineScreen.Instance.DisableInput();
			NTimelineScreen.Instance.OpenInspectScreen(this, playAnimation: false);
		}
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", Vector2.One * 1.05f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
		_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_s), 1f, 0.05);
	}

	private void RevealEpoch()
	{
		State = EpochSlotState.Complete;
		_portrait.Visible = true;
		_portrait.Texture = model.Portrait;
		DisableHighlight();
		SaveManager.Instance.RevealEpoch(model.Id);
		_slotImage.Modulate = Colors.White;
		_slotImage.ClipChildren = ClipChildrenMode.AndDraw;
		_chains.Visible = false;
		NTimelineScreen.Instance.OpenInspectScreen(this, playAnimation: true);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_isGlowPulsing = false;
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		if (NControllerManager.Instance.IsUsingController)
		{
			_selectionReticle.OnSelect();
		}
		if (State != EpochSlotState.NotObtained)
		{
			_hoverTween.TweenProperty(this, "scale", Vector2.One * 1.05f, 0.05);
			if (State == EpochSlotState.Complete)
			{
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 1.1f, 0.05);
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1.1f, 0.05);
				LocString unlockInfo = model.UnlockInfo;
				unlockInfo.Add("IsRevealed", variable: true);
				_hoverTip = new HoverTip(model.Title, unlockInfo, PreloadManager.Cache.GetTexture2D("res://images/packed/unlock_icon.png"));
			}
			else if (State == EpochSlotState.Obtained)
			{
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.65f, 0.05);
				LocString unlockInfo2 = model.UnlockInfo;
				unlockInfo2.Add("IsRevealed", variable: false);
				_hoverTip = new HoverTip(model.Title, unlockInfo2);
			}
		}
		else
		{
			_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.25f, 0.05);
			_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1.2f, 0.05);
			LocString unlockInfo3 = model.UnlockInfo;
			unlockInfo3.Add("IsRevealed", variable: false);
			_hoverTip = new HoverTip(model.Title, unlockInfo3);
		}
		if (_hoverTip != null)
		{
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
			float num = (base.Size.X * 0.5f + 6f) * GetGlobalTransform().Scale.X;
			float num2 = base.GlobalPosition.X + (base.Size.X * 0.5f + 6f) * GetGlobalTransform().Scale.X;
			if (base.GlobalPosition.X > NGame.Instance.GetViewportRect().Size.X * 0.7f)
			{
				nHoverTipSet.GlobalPosition = new Vector2(num2 - num - 360f, base.GlobalPosition.Y);
			}
			else
			{
				nHoverTipSet.GlobalPosition = new Vector2(num2 + num, base.GlobalPosition.Y);
			}
			nHoverTipSet.SetFollowOwner();
		}
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_selectionReticle.OnDeselect();
		_hoverTween?.Kill();
		_hoverTween = CreateTween().SetParallel();
		_hoverTween.TweenProperty(this, "scale", Vector2.One, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		if (State != EpochSlotState.NotObtained)
		{
			if (State == EpochSlotState.Obtained)
			{
				_isGlowPulsing = true;
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.5f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			}
			else if (State == EpochSlotState.Complete)
			{
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			}
		}
		else
		{
			_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 0.25f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1.1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
		NHoverTipSet.Remove(this);
	}

	protected override void OnPress()
	{
		if (State != EpochSlotState.NotObtained)
		{
			base.OnPress();
			_hoverTween?.Kill();
			_hoverTween = CreateTween().SetParallel();
			_hoverTween.TweenProperty(this, "scale", Vector2.One * 0.95f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			if (State == EpochSlotState.Complete)
			{
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderS), _hsv.GetShaderParameter(_s), 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 1f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			}
			else if (State == EpochSlotState.Obtained)
			{
				_hoverTween.TweenMethod(Callable.From<float>(UpdateShaderV), _hsv.GetShaderParameter(_v), 0.5f, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			}
		}
	}

	public async Task SpawnSlot()
	{
		_spawnTween = CreateTween().SetParallel();
		_spawnTween.Chain();
		_spawnTween.TweenInterval(Rng.Chaotic.NextDouble(0.0, 0.3));
		_spawnTween.Chain();
		_spawnTween.TweenProperty(_slotImage, "position", Vector2.Zero, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(new Vector2(0f, 64f));
		_spawnTween.TweenProperty(_outline, "position", Vector2.Zero, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back)
			.From(new Vector2(0f, 64f));
		_spawnTween.TweenProperty(_slotImage, "self_modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		await ToSignal(_spawnTween, Tween.SignalName.Finished);
		HasSpawned = true;
		if (State == EpochSlotState.Obtained)
		{
			EnableHighlight();
		}
		else
		{
			_outline.Modulate = _defaultSlotOutlineColor;
		}
	}

	public override void _Process(double delta)
	{
		if (_isGlowPulsing)
		{
			_outline.Modulate = new Color(_outline.Modulate.R, _outline.Modulate.G, _outline.Modulate.B, (Mathf.Sin((float)Time.GetTicksMsec() * 0.005f) + 2f) * 0.25f);
		}
	}

	private void DisableHighlight()
	{
		_isGlowPulsing = false;
		_outline.Modulate = Colors.Transparent;
		_highlightVfx?.QueueFreeSafely();
		_offscreenVfx?.QueueFreeSafely();
	}

	private void EnableHighlight()
	{
		_isGlowPulsing = true;
		_outline.Modulate = _highlightSlotColor;
		_outline.SelfModulate = StsColors.transparentWhite;
		_glowTween?.Kill();
		_glowTween = CreateTween();
		_glowTween.TweenProperty(_outline, "self_modulate:a", 1f, 1.0);
		_offscreenVfx = NEpochOffscreenVfx.Create(this);
		_highlightVfx = NEpochHighlightVfx.Create();
		this.AddChildSafely(_highlightVfx);
		NTimelineScreen.Instance.GetReminderVfxHolder().AddChildSafely(_offscreenVfx);
		MoveChild(_highlightVfx, 0);
	}

	public void SetState(EpochSlotState setState)
	{
		State = setState;
		if (State == EpochSlotState.None)
		{
			Log.Error("Slot State is invalid.");
			return;
		}
		_slotImage.Modulate = Colors.White;
		_slotImage.ClipChildren = ClipChildrenMode.AndDraw;
		_portrait.Visible = true;
		if (State == EpochSlotState.Complete)
		{
			DisableHighlight();
			_portrait.Texture = model.Portrait;
			UpdateShaderS(1f);
			UpdateShaderV(1f);
		}
		else if (State == EpochSlotState.Obtained)
		{
			_portrait.Texture = model.Portrait;
			UpdateShaderS(0f);
			UpdateShaderV(0.5f);
			_chains.Visible = true;
		}
		else if (State == EpochSlotState.NotObtained)
		{
			_blurShader.SetShaderParameter(_lod, 2f);
			UpdateShaderS(0.25f);
			UpdateShaderV(1f);
			_blurPortrait.Texture = model.Portrait;
			_subViewportContainer.Visible = true;
			_portrait.Texture = _subViewport.GetTexture();
		}
		base.MouseDefaultCursorShape = (CursorShape)((State == EpochSlotState.Complete) ? 16 : 0);
	}

	private void UpdateShaderS(float value)
	{
		_hsv.SetShaderParameter(_s, value);
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}

	private void UpdateBlurLod(float value)
	{
		_blurShader.SetShaderParameter(_lod, value);
	}
}
