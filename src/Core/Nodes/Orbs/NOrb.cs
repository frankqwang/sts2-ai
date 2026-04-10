using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Orbs;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;

namespace MegaCrit.Sts2.Core.Nodes.Orbs;

public partial class NOrb : NClickableControl
{
	private TextureRect _outline;

	private Control _visualContainer;

	private Control _labelContainer;

	private MegaLabel _passiveLabel;

	private MegaLabel _evokeLabel;

	private Control _bounds;

	private CpuParticles2D _flashParticle;

	private NSelectionReticle _selectionReticle;

	private bool _isLocal;

	private Node2D? _sprite;

	private Tween? _curTween;

	public OrbModel? Model { get; private set; }

	private static string ScenePath => SceneHelper.GetScenePath("/orbs/orb");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public static NOrb Create(bool isLocal)
	{
		NOrb nOrb = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NOrb>(PackedScene.GenEditState.Disabled);
		nOrb._isLocal = isLocal;
		return nOrb;
	}

	public static NOrb Create(bool isLocal, OrbModel? model)
	{
		NOrb nOrb = Create(isLocal);
		nOrb.Model = model;
		return nOrb;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_outline = GetNode<TextureRect>("%Outline");
		_visualContainer = GetNode<Control>("%VisualContainer");
		_passiveLabel = GetNode<MegaLabel>("%PassiveAmount");
		_evokeLabel = GetNode<MegaLabel>("%EvokeAmount");
		_flashParticle = GetNode<CpuParticles2D>("%Flash");
		_bounds = GetNode<Control>("Bounds");
		_labelContainer = GetNode<Control>("%LabelContainer");
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		if (Model != null)
		{
			CreateTween().TweenProperty(_outline, "scale", Vector2.One, 0.25).From(Vector2.Zero);
		}
		if (_isLocal)
		{
			base.Scale *= 0.85f;
		}
		UpdateVisuals(isEvoking: false);
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		if (Model != null)
		{
			Model.Triggered += Flash;
		}
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		if (Model != null)
		{
			Model.Triggered -= Flash;
		}
	}

	public void ReplaceOrb(OrbModel model)
	{
		_sprite?.QueueFreeSafely();
		_sprite = null;
		Model = model;
		UpdateVisuals(isEvoking: false);
	}

	public void UpdateVisuals(bool isEvoking)
	{
		if (!IsNodeReady() || !CombatManager.Instance.IsInProgress)
		{
			return;
		}
		if (Model == null)
		{
			_sprite?.QueueFreeSafely();
			_passiveLabel.Visible = false;
			_evokeLabel.Visible = false;
			_outline.Visible = _isLocal;
			_flashParticle.Visible = false;
			return;
		}
		if (_sprite == null)
		{
			_sprite = Model.CreateSprite();
			_visualContainer.AddChildSafely(_sprite);
			_sprite.Position = Vector2.Zero;
			_curTween?.Kill();
			_curTween = CreateTween();
			_curTween.TweenProperty(_sprite, "scale", Vector2.One, 0.5).From(Vector2.Zero).SetTrans(Tween.TransitionType.Back)
				.SetEase(Tween.EaseType.Out);
		}
		_outline.Visible = false;
		_flashParticle.Visible = true;
		_flashParticle.Texture = Model.Icon;
		_labelContainer.Visible = _isLocal;
		if (!_isLocal)
		{
			base.Modulate = Model.DarkenedColor;
		}
		OrbModel model = Model;
		if (!(model is PlasmaOrb))
		{
			if (!(model is DarkOrb))
			{
				if (model is GlassOrb)
				{
					_passiveLabel.Visible = !isEvoking;
					_evokeLabel.Visible = isEvoking;
					_sprite.Modulate = ((Model.PassiveVal == 0m) ? Model.DarkenedColor : Colors.White);
					_passiveLabel.SetTextAutoSize(Model.PassiveVal.ToString("0"));
					_evokeLabel.SetTextAutoSize(Model.EvokeVal.ToString("0"));
				}
				else
				{
					_passiveLabel.Visible = !isEvoking;
					_evokeLabel.Visible = isEvoking;
					_passiveLabel.SetTextAutoSize(Model.PassiveVal.ToString("0"));
					_evokeLabel.SetTextAutoSize(Model.EvokeVal.ToString("0"));
				}
			}
			else
			{
				_passiveLabel.Visible = true;
				_evokeLabel.Visible = true;
				_passiveLabel.SetTextAutoSize(Model.PassiveVal.ToString("0"));
				_evokeLabel.SetTextAutoSize(Model.EvokeVal.ToString("0"));
			}
		}
		else
		{
			_passiveLabel.Visible = false;
			_evokeLabel.Visible = false;
		}
	}

	private void Flash()
	{
		_flashParticle.Emitting = true;
	}

	protected override void OnFocus()
	{
		if (Model != null || _isLocal)
		{
			IEnumerable<IHoverTip> enumerable;
			if (Model != null)
			{
				enumerable = Model.HoverTips;
			}
			else
			{
				IEnumerable<IHoverTip> enumerable2 = new List<IHoverTip> { OrbModel.EmptySlotHoverTipHoverTip };
				enumerable = enumerable2;
			}
			IEnumerable<IHoverTip> hoverTips = enumerable;
			NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(_bounds, hoverTips, HoverTip.GetHoverTipAlignment(_bounds));
			nHoverTipSet.SetFollowOwner();
			_labelContainer.Visible = true;
			base.Modulate = Colors.White;
			if (NControllerManager.Instance.IsUsingController)
			{
				_selectionReticle.OnSelect();
			}
		}
	}

	protected override void OnUnfocus()
	{
		_labelContainer.Visible = _isLocal;
		if (Model != null)
		{
			base.Modulate = (_isLocal ? Colors.White : Model.DarkenedColor);
		}
		NHoverTipSet.Remove(_bounds);
		_selectionReticle.OnDeselect();
	}
}
