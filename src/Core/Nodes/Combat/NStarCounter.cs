using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NStarCounter : Control
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly string _starGainVfxPath = SceneHelper.GetScenePath("vfx/star_gain_vfx");

	private Player? _player;

	private MegaRichTextLabel _label;

	private Control _rotationLayers;

	private Control _icon;

	private ShaderMaterial _hsv;

	private float _lerpingStarCount;

	private float _velocity;

	private int _displayedStarCount;

	private Tween? _hsvTween;

	private bool _isListeningToCombatState;

	private HoverTip _hoverTip;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_starGainVfxPath);

	public override void _Ready()
	{
		_label = GetNode<MegaRichTextLabel>("%CountLabel");
		_rotationLayers = GetNode<Control>("%RotationLayers");
		_icon = GetNode<Control>("Icon");
		_hsv = (ShaderMaterial)_icon.Material;
		LocString locString = new LocString("static_hover_tips", "STAR_COUNT.description");
		locString.Add("singleStarIcon", "[img]res://images/packed/sprite_fonts/star_icon.png[/img]");
		_hoverTip = new HoverTip(new LocString("static_hover_tips", "STAR_COUNT.title"), locString);
		Connect(Control.SignalName.MouseEntered, Callable.From(OnHovered));
		Connect(Control.SignalName.MouseExited, Callable.From(OnUnhovered));
		base.Visible = false;
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		ConnectStarsChangedSignal();
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (_player != null && _isListeningToCombatState)
		{
			_player.PlayerCombatState.StarsChanged -= OnStarsChanged;
			_isListeningToCombatState = false;
		}
	}

	private void ConnectStarsChangedSignal()
	{
		if (_player != null && !_isListeningToCombatState)
		{
			_player.PlayerCombatState.StarsChanged += OnStarsChanged;
			_isListeningToCombatState = true;
		}
	}

	public void Initialize(Player player)
	{
		_player = player;
		ConnectStarsChangedSignal();
		RefreshVisibility();
	}

	private void OnHovered()
	{
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _hoverTip);
		nHoverTipSet.GlobalPosition = base.GlobalPosition + new Vector2(-34f, -300f);
	}

	private void OnUnhovered()
	{
		NHoverTipSet.Remove(this);
	}

	private void OnStarsChanged(int oldStars, int newStars)
	{
		UpdateStarCount(oldStars, newStars);
		RefreshVisibility();
	}

	public override void _Process(double delta)
	{
		if (_player != null)
		{
			float num = ((_player.PlayerCombatState.Stars == 0) ? 5f : 30f);
			for (int i = 0; i < _rotationLayers.GetChildCount(); i++)
			{
				_rotationLayers.GetChild<Control>(i).RotationDegrees += (float)delta * num * (float)(i + 1);
			}
			_lerpingStarCount = MathHelper.SmoothDamp(_lerpingStarCount, _player.PlayerCombatState.Stars, ref _velocity, 0.1f, (float)delta);
			SetStarCountText(Mathf.RoundToInt(_lerpingStarCount));
		}
	}

	private void UpdateStarCount(int oldCount, int newCount)
	{
		if (newCount < oldCount)
		{
			_hsvTween?.Kill();
			_hsv.SetShaderParameter(_v, 1f);
			_lerpingStarCount = newCount;
			SetStarCountText(newCount);
		}
		else if (newCount > oldCount)
		{
			_hsvTween?.Kill();
			_hsvTween = CreateTween();
			_hsvTween.TweenMethod(Callable.From<float>(UpdateShaderV), 2f, 1f, 0.20000000298023224);
			Node2D node2D = PreloadManager.Cache.GetAsset<PackedScene>(_starGainVfxPath).Instantiate<Node2D>(PackedScene.GenEditState.Disabled);
			this.AddChildSafely(node2D);
			MoveChild(node2D, 0);
			node2D.Position = base.Size / 2f;
		}
	}

	private void SetStarCountText(int stars)
	{
		if (_displayedStarCount != stars)
		{
			_displayedStarCount = stars;
			_label.AddThemeColorOverride(ThemeConstants.Label.fontColor, (stars == 0) ? StsColors.red : StsColors.cream);
			_label.Text = $"[center]{stars}[/center]";
			if (stars == 0)
			{
				_hsv.SetShaderParameter(_s, 0.5f);
				_hsv.SetShaderParameter(_v, 0.85f);
			}
			else
			{
				_hsv.SetShaderParameter(_s, 1f);
				_hsv.SetShaderParameter(_v, 1f);
			}
		}
	}

	private void UpdateShaderV(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}

	private void RefreshVisibility()
	{
		if (_player == null)
		{
			base.Visible = false;
			return;
		}
		int stars = _player.PlayerCombatState.Stars;
		base.Visible = base.Visible || _player.Character.ShouldAlwaysShowStarCounter || stars > 0;
	}
}
