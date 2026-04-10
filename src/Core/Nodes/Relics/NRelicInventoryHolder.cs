using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Relics;

public partial class NRelicInventoryHolder : NButton
{
	private static readonly string _scenePath = SceneHelper.GetScenePath("relics/relic_inventory_holder");

	private static readonly string _flashPath = SceneHelper.GetScenePath("vfx/relic_inventory_flash_vfx");

	private const float _newlyAcquiredPopDuration = 0.35f;

	private const float _newlyAcquiredFadeInDuration = 0.1f;

	private const float _newlyAcquiredPopDistance = 40f;

	private NRelic _relic;

	private RelicModel? _subscribedRelic;

	private MegaLabel _amountLabel;

	private Tween? _hoverTween;

	private Tween? _obtainedTween;

	private CancellationTokenSource? _cancellationTokenSource;

	private Vector2 _originalIconPosition;

	private RelicModel _model;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { _scenePath, _flashPath });

	public NRelic Relic => _relic;

	public NRelicInventory Inventory { get; set; }

	public static NRelicInventoryHolder? Create(RelicModel relic)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NRelicInventoryHolder nRelicInventoryHolder = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRelicInventoryHolder>(PackedScene.GenEditState.Disabled);
		nRelicInventoryHolder.Name = $"NRelicContainerHolder-{relic.Id}";
		nRelicInventoryHolder._model = relic;
		return nRelicInventoryHolder;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_relic = GetNode<NRelic>("%Relic");
		_amountLabel = GetNode<MegaLabel>("%AmountLabel");
		_originalIconPosition = _relic.Icon.Position;
		_relic.ModelChanged += OnModelChanged;
		_relic.Model = _model;
	}

	public override void _ExitTree()
	{
		_hoverTween?.Kill();
		if (_subscribedRelic != null)
		{
			_subscribedRelic.DisplayAmountChanged -= OnDisplayAmountChanged;
			_subscribedRelic.StatusChanged -= OnStatusChanged;
			_subscribedRelic.Flashed -= OnRelicFlashed;
		}
		_subscribedRelic = null;
		_relic.ModelChanged -= OnModelChanged;
	}

	private void OnModelChanged(RelicModel? oldModel, RelicModel? newModel)
	{
		if (oldModel != null)
		{
			oldModel.DisplayAmountChanged -= OnDisplayAmountChanged;
			oldModel.StatusChanged -= OnStatusChanged;
			oldModel.Flashed -= OnRelicFlashed;
		}
		if (newModel != null)
		{
			newModel.DisplayAmountChanged += OnDisplayAmountChanged;
			newModel.StatusChanged += OnStatusChanged;
			newModel.Flashed += OnRelicFlashed;
		}
		RefreshAmount();
		RefreshStatus();
		_subscribedRelic = newModel;
	}

	private void RefreshAmount()
	{
		if (_relic.Model.ShowCounter && RunManager.Instance.IsInProgress)
		{
			_amountLabel.Visible = true;
			_amountLabel.SetTextAutoSize(_relic.Model.DisplayAmount.ToString());
		}
		else
		{
			_amountLabel.Visible = false;
		}
	}

	private void RefreshStatus()
	{
		if (!RunManager.Instance.IsInProgress)
		{
			_relic.Icon.Modulate = Colors.White;
			return;
		}
		_relic.Model.UpdateTexture(_relic.Icon);
		TextureRect icon = _relic.Icon;
		Color modulate;
		switch (_relic.Model.Status)
		{
		case RelicStatus.Normal:
		case RelicStatus.Active:
			modulate = Colors.White;
			break;
		case RelicStatus.Disabled:
			modulate = new Color("#808080");
			break;
		default:
			throw new ArgumentOutOfRangeException();
		}
		icon.Modulate = modulate;
	}

	public async Task PlayNewlyAcquiredAnimation(Vector2? startLocation, Vector2? startScale)
	{
		if (_cancellationTokenSource != null)
		{
			await _cancellationTokenSource.CancelAsync();
		}
		CancellationTokenSource cancelTokenSource = (_cancellationTokenSource = new CancellationTokenSource());
		await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		if (!cancelTokenSource.IsCancellationRequested)
		{
			_obtainedTween?.Kill();
			if (!startLocation.HasValue)
			{
				TextureRect icon = _relic.Icon;
				Vector2 position = _relic.Icon.Position;
				position.Y = _relic.Icon.Position.Y + 40f;
				icon.Position = position;
				TextureRect icon2 = _relic.Icon;
				Color modulate = _relic.Icon.Modulate;
				modulate.A = 0f;
				icon2.Modulate = modulate;
				_obtainedTween = GetTree().CreateTween();
				_obtainedTween.TweenProperty(_relic.Icon, "modulate:a", 1f, 0.10000000149011612);
				_obtainedTween.Parallel();
				_obtainedTween.SetEase(Tween.EaseType.Out);
				_obtainedTween.SetTrans(Tween.TransitionType.Back);
				_obtainedTween.TweenProperty(_relic.Icon, "position:y", _originalIconPosition.Y, 0.3499999940395355);
				_obtainedTween.TweenCallback(Callable.From(DoFlash));
			}
			else
			{
				_relic.Icon.GlobalPosition = startLocation.Value;
				_relic.Icon.Scale = startScale ?? Vector2.One;
				TextureRect icon3 = _relic.Icon;
				Color modulate = _relic.Icon.Modulate;
				modulate.A = 1f;
				icon3.Modulate = modulate;
				_obtainedTween = GetTree().CreateTween();
				_obtainedTween.SetEase(Tween.EaseType.Out);
				_obtainedTween.SetTrans(Tween.TransitionType.Sine);
				_obtainedTween.TweenProperty(_relic.Icon, "position", _originalIconPosition, 0.3499999940395355);
				_obtainedTween.Parallel().TweenProperty(_relic.Icon, "scale", Vector2.One, 0.3499999940395355);
				_obtainedTween.TweenCallback(Callable.From(DoFlash));
			}
		}
	}

	protected override void OnFocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relic.Icon, "scale", Vector2.One * 1.25f, 0.05);
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(this, _relic.Model.HoverTips);
		nHoverTipSet.SetAlignmentForRelic(_relic);
	}

	protected override void OnUnfocus()
	{
		_hoverTween?.Kill();
		_hoverTween = CreateTween();
		_hoverTween.TweenProperty(_relic.Icon, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		NHoverTipSet.Remove(this);
	}

	private void OnRelicFlashed(RelicModel _, IEnumerable<Creature> __)
	{
		DoFlash();
	}

	private void DoFlash()
	{
		Node2D node2D = PreloadManager.Cache.GetScene(_flashPath).Instantiate<Node2D>(PackedScene.GenEditState.Disabled);
		Node aboveTopBarVfxContainer = NRun.Instance.GlobalUi.AboveTopBarVfxContainer;
		node2D.GetNode<GpuParticles2D>("Particles").Texture = _relic.Model.Icon;
		node2D.GlobalPosition = base.GlobalPosition + base.Size * 0.5f;
		aboveTopBarVfxContainer.AddChildSafely(node2D);
	}

	private void OnDisplayAmountChanged()
	{
		RefreshAmount();
	}

	private void OnStatusChanged()
	{
		RefreshStatus();
	}
}
