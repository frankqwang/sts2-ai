using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Entities.UI;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;

namespace MegaCrit.Sts2.Core.Nodes.Screens.CharacterSelect;

public partial class NAscensionPanel : Control
{
	[Signal]
	public delegate void AscensionLevelChangedEventHandler();

	private static readonly StringName _tabLeftHotkey = MegaInput.viewDeckAndTabLeft;

	private static readonly StringName _tabRightHotkey = MegaInput.viewExhaustPileAndTabRight;

	private static readonly StringName _fontOutlineTheme = "font_outline_color";

	private static readonly StringName _h = new StringName("h");

	private static readonly StringName _v = new StringName("v");

	private static readonly Color _redLabelOutline = new Color("593400");

	private static readonly Color _blueLabelOutline = new Color("004759");

	private int _maxAscension;

	private NButton _leftArrow;

	private NButton _rightArrow;

	private MegaLabel _ascensionLevel;

	private MegaRichTextLabel _info;

	private TextureRect _leftTriggerIcon;

	private TextureRect _rightTriggerIcon;

	private ShaderMaterial _iconHsv;

	private bool _arrowsVisible = true;

	private MultiplayerUiMode _mode = MultiplayerUiMode.Singleplayer;

	private Tween? _tween;

	public int Ascension { get; private set; }

	public override void _Ready()
	{
		_leftTriggerIcon = GetNode<TextureRect>("%LeftTriggerIcon");
		_rightTriggerIcon = GetNode<TextureRect>("%RightTriggerIcon");
		_leftArrow = GetNode<NButton>("HBoxContainer/LeftArrowContainer/LeftArrow");
		_rightArrow = GetNode<NButton>("HBoxContainer/RightArrowContainer/RightArrow");
		_ascensionLevel = GetNode<MegaLabel>("HBoxContainer/AscensionIconContainer/AscensionIcon/AscensionLevel");
		_info = GetNode<MegaRichTextLabel>("HBoxContainer/AscensionDescription/Description");
		_iconHsv = (ShaderMaterial)GetNode<Control>("%AscensionIcon").Material;
		_leftArrow.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			DecrementAscension();
		}));
		_rightArrow.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
		{
			IncrementAscension();
		}));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.MouseDetected, Callable.From(UpdateControllerButton));
		NControllerManager.Instance.Connect(NControllerManager.SignalName.ControllerDetected, Callable.From(UpdateControllerButton));
		NInputManager.Instance.Connect(NInputManager.SignalName.InputRebound, Callable.From(UpdateControllerButton));
		UpdateControllerButton();
	}

	public void Initialize(MultiplayerUiMode mode)
	{
		_mode = mode;
		if (_mode == MultiplayerUiMode.Host)
		{
			SetFireBlue();
			_arrowsVisible = true;
			SetMaxAscension(SaveManager.Instance.Progress.MaxMultiplayerAscension);
			SetAscensionLevel(Math.Min(_maxAscension, SaveManager.Instance.Progress.PreferredMultiplayerAscension));
			NHotkeyManager.Instance.PushHotkeyPressedBinding(_tabLeftHotkey, DecrementAscension);
			NHotkeyManager.Instance.PushHotkeyPressedBinding(_tabRightHotkey, IncrementAscension);
		}
		else if (_mode == MultiplayerUiMode.Singleplayer)
		{
			SetFireRed();
			_arrowsVisible = true;
			SetMaxAscension(0);
			SetAscensionLevel(0);
			NHotkeyManager.Instance.PushHotkeyPressedBinding(_tabLeftHotkey, DecrementAscension);
			NHotkeyManager.Instance.PushHotkeyPressedBinding(_tabRightHotkey, IncrementAscension);
		}
		else
		{
			MultiplayerUiMode mode2 = _mode;
			if ((uint)(mode2 - 3) <= 1u)
			{
				SetFireBlue();
				_arrowsVisible = false;
				SetMaxAscension(0);
			}
		}
	}

	private void SetFireBlue()
	{
		_iconHsv.SetShaderParameter(_h, 0.52f);
		_iconHsv.SetShaderParameter(_v, 1.2f);
		_ascensionLevel.AddThemeColorOverride(_fontOutlineTheme, _blueLabelOutline);
	}

	private void SetFireRed()
	{
		_iconHsv.SetShaderParameter(_h, 1f);
		_iconHsv.SetShaderParameter(_v, 1f);
		_ascensionLevel.AddThemeColorOverride(_fontOutlineTheme, _redLabelOutline);
	}

	public void Cleanup()
	{
		MultiplayerUiMode mode = _mode;
		if ((uint)(mode - 1) <= 1u)
		{
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(_tabLeftHotkey, DecrementAscension);
			NHotkeyManager.Instance.RemoveHotkeyPressedBinding(_tabRightHotkey, IncrementAscension);
		}
	}

	public void SetAscensionLevel(int ascension)
	{
		if (Ascension != ascension)
		{
			Ascension = ascension;
			EmitSignal(SignalName.AscensionLevelChanged);
		}
		RefreshAscensionText();
		RefreshArrowVisibility();
	}

	private void IncrementAscension()
	{
		if (Ascension < _maxAscension)
		{
			SetAscensionLevel(Ascension + 1);
		}
	}

	private void DecrementAscension()
	{
		if (Ascension > 0)
		{
			SetAscensionLevel(Ascension - 1);
		}
	}

	private void RefreshArrowVisibility()
	{
		_leftArrow.Visible = _arrowsVisible && Ascension != 0;
		_rightArrow.Visible = _arrowsVisible && Ascension != _maxAscension;
	}

	public void SetMaxAscension(int maxAscension)
	{
		Log.Info($"Max ascension changed to {maxAscension}");
		_maxAscension = maxAscension;
		if (Ascension >= _maxAscension)
		{
			SetAscensionLevel(_maxAscension);
		}
		base.Visible = _maxAscension > 0;
		RefreshArrowVisibility();
	}

	private void RefreshAscensionText()
	{
		_ascensionLevel.SetTextAutoSize(Ascension.ToString());
		string formattedText = AscensionHelper.GetTitle(Ascension).GetFormattedText();
		string formattedText2 = AscensionHelper.GetDescription(Ascension).GetFormattedText();
		_info.Text = "[b][gold]" + formattedText + "[/gold][/b]\n" + formattedText2;
	}

	public void AnimIn()
	{
		if (base.Visible)
		{
			Color modulate = base.Modulate;
			modulate.A = 0f;
			base.Modulate = modulate;
			_tween?.FastForwardToCompletion();
			_tween = CreateTween().SetParallel();
			_tween.TweenProperty(this, "modulate:a", 1f, 0.2);
			_tween.TweenProperty(this, "position:y", base.Position.Y, 0.3).From(base.Position.Y + 30f).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Back);
		}
	}

	private void UpdateControllerButton()
	{
		MultiplayerUiMode mode = _mode;
		if ((uint)(mode - 1) <= 1u)
		{
			_leftTriggerIcon.Visible = NControllerManager.Instance.IsUsingController;
			_rightTriggerIcon.Visible = NControllerManager.Instance.IsUsingController;
			_leftTriggerIcon.Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.viewDeckAndTabLeft);
			_rightTriggerIcon.Texture = NInputManager.Instance.GetHotkeyIcon(MegaInput.viewExhaustPileAndTabRight);
		}
		else
		{
			_leftTriggerIcon.Visible = false;
			_rightTriggerIcon.Visible = false;
		}
	}
}
