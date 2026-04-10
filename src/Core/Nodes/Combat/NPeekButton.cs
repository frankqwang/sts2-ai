using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Audio.Debug;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.ControllerInput;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Screens.Capstones;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NPeekButton : NButton
{
	[Signal]
	public delegate void ToggledEventHandler(NPeekButton peekButton);

	private static readonly StringName _pulseStrength = new StringName("pulse_strength");

	private readonly List<Control> _targets = new List<Control>();

	private readonly List<Control> _hiddenTargets = new List<Control>();

	private TextureRect _flash;

	private Control _visuals;

	private IOverlayScreen? _overlayScreenParent;

	private Tween? _hoverTween;

	private Tween? _wiggleTween;

	protected override string[] Hotkeys => new string[1] { MegaInput.peek };

	public bool IsPeeking { get; private set; }

	public Marker2D CurrentCardMarker { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		_flash = GetNode<TextureRect>("%Flash");
		_visuals = GetNode<Control>("%Visuals");
		CurrentCardMarker = GetNode<Marker2D>("%CurrentCardMarker");
		if (NCombatRoom.Instance != null)
		{
			if (NCombatRoom.Instance.IsNodeReady())
			{
				OnCombatRoomReady();
			}
			else
			{
				NCombatRoom.Instance.Connect(Node.SignalName.Ready, Callable.From(OnCombatRoomReady));
			}
		}
		if (!CombatManager.Instance.IsInProgress)
		{
			Disable();
		}
		for (Node parent = GetParent(); parent != null; parent = parent.GetParent())
		{
			if (parent is IOverlayScreen overlayScreenParent)
			{
				_overlayScreenParent = overlayScreenParent;
				NOverlayStack.Instance?.Connect(NOverlayStack.SignalName.Changed, Callable.From(OnOverlayStackChanged));
				NCapstoneContainer.Instance?.Connect(NCapstoneContainer.SignalName.Changed, Callable.From(OnOverlayStackChanged));
				break;
			}
		}
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		base.Visible = true;
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		base.Visible = false;
	}

	private void OnOverlayStackChanged()
	{
		if (IsPeeking && _overlayScreenParent != null && NCapstoneContainer.Instance?.CurrentCapstoneScreen == null && _overlayScreenParent == NOverlayStack.Instance?.Peek())
		{
			NOverlayStack.Instance.HideBackstop();
		}
	}

	public void Wiggle()
	{
		_flash.Visible = true;
		TextureRect flash = _flash;
		Color modulate = _flash.Modulate;
		modulate.A = 0f;
		flash.Modulate = modulate;
		_wiggleTween?.Kill();
		_wiggleTween = CreateTween();
		_visuals.RotationDegrees = 0f;
		_wiggleTween.TweenMethod(Callable.From(delegate(float t)
		{
			_visuals.RotationDegrees = 10f * Mathf.Sin(t * 3f) * Mathf.Sin(t * 0.5f);
		}), 0f, (float)Math.PI * 2f, 0.5);
		_wiggleTween.Parallel().TweenMethod(Callable.From(delegate(float t)
		{
			_visuals.Scale = Vector2.One + Vector2.One * 0.15f * Mathf.Sin(t) * Mathf.Sin(t * 0.5f);
		}), 0f, (float)Math.PI, 0.25);
		_wiggleTween.Parallel().TweenProperty(_flash, "modulate:a", 1f, 0.1);
		_wiggleTween.Chain().TweenProperty(_flash, "modulate:a", 0f, 0.3).SetTrans(Tween.TransitionType.Quad)
			.SetEase(Tween.EaseType.Out);
		NDebugAudioManager.Instance.Play("deny.mp3", 0.5f, PitchVariance.Medium);
	}

	public void AddTargets(params Control[] targets)
	{
		_targets.AddRange(targets);
	}

	public void SetPeeking(bool isPeeking)
	{
		if (IsPeeking == isPeeking)
		{
			return;
		}
		IsPeeking = isPeeking;
		if (NOverlayStack.Instance.ScreenCount > 0)
		{
			if (IsPeeking)
			{
				NOverlayStack.Instance.HideBackstop();
			}
			else
			{
				NOverlayStack.Instance.ShowBackstop();
			}
		}
		if (IsPeeking)
		{
			foreach (Control item in _targets.Where((Control t) => t.Visible))
			{
				_hiddenTargets.Add(item);
				item.Visible = false;
			}
		}
		else
		{
			foreach (Control hiddenTarget in _hiddenTargets)
			{
				hiddenTarget.Visible = true;
			}
			_hiddenTargets.Clear();
		}
		((ShaderMaterial)_visuals.Material).SetShaderParameter(_pulseStrength, IsPeeking ? 1 : 0);
		EmitSignal(SignalName.Toggled, this);
	}

	protected override void OnRelease()
	{
		SetPeeking(!IsPeeking);
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_visuals, "scale", Vector2.One, 0.15);
	}

	protected override void OnPress()
	{
		base.OnPress();
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_visuals, "scale", Vector2.One * 0.95f, 0.05);
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_visuals, "scale", Vector2.One * 1.1f, 0.05);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_visuals, "scale", Vector2.One, 0.15);
	}

	private void OnCombatRoomReady()
	{
		NCombatRoom.Instance.Ui.OnPeekButtonReady(this);
	}
}
