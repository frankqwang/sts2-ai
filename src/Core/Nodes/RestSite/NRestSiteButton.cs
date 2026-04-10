using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Extensions;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Runs;

namespace MegaCrit.Sts2.Core.Nodes.RestSite;

public partial class NRestSiteButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private ShaderMaterial _hsv;

	private Control _visuals;

	private TextureRect _icon;

	private Control _outline;

	private MegaLabel _label;

	private Vector2 _labelPosition;

	private bool _isUnclickable;

	private bool _executingOption;

	private const double _unfocusAnimDur = 1.0;

	private readonly CancellationTokenSource _cts = new CancellationTokenSource();

	private Tween? _currentTween;

	private RestSiteOption? _option;

	private static readonly string _scenePath = SceneHelper.GetScenePath("rest_site/rest_site_button");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public RestSiteOption Option
	{
		get
		{
			return _option ?? throw new InvalidOperationException("Option accessed before being set");
		}
		set
		{
			_option = value;
			Reload();
		}
	}

	public override void _Ready()
	{
		ConnectSignals();
		base.Modulate = StsColors.transparentBlack;
		_visuals = GetNode<Control>("%Visuals");
		_icon = GetNode<TextureRect>("%Icon");
		_outline = GetNode<Control>("%Outline");
		_label = GetNode<MegaLabel>("%Label");
		_hsv = (ShaderMaterial)_icon.Material;
		_labelPosition = _label.Position;
		TaskHelper.RunSafely(AnimateIn());
		Reload();
	}

	private async Task AnimateIn()
	{
		Tween tween = CreateTween();
		tween.TweenProperty(this, "modulate", Colors.White, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Cubic);
		await tween.AwaitFinished(_cts.Token);
		if (this.IsValid())
		{
			base.MouseFilter = MouseFilterEnum.Stop;
		}
	}

	public override void _ExitTree()
	{
		_cts.Cancel();
		_cts.Dispose();
		_currentTween?.Kill();
	}

	public static NRestSiteButton Create(RestSiteOption option)
	{
		NRestSiteButton nRestSiteButton = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NRestSiteButton>(PackedScene.GenEditState.Disabled);
		nRestSiteButton.Option = option;
		nRestSiteButton._isUnclickable = !option.IsEnabled;
		return nRestSiteButton;
	}

	private void Reload()
	{
		if (IsNodeReady() && !(_option == null))
		{
			_icon.Texture = Option.Icon;
			_label.SetTextAutoSize(Option.Title.GetFormattedText());
			if (!_option.IsEnabled)
			{
				_hsv.SetShaderParameter(_s, 0f);
				_hsv.SetShaderParameter(_v, 0.6f);
			}
			else
			{
				_hsv.SetShaderParameter(_s, 1f);
				_hsv.SetShaderParameter(_v, 1f);
			}
		}
	}

	protected override void OnRelease()
	{
		if (!_isUnclickable)
		{
			base.OnRelease();
			TaskHelper.RunSafely(SelectOption(Option));
		}
	}

	private async Task SelectOption(RestSiteOption option)
	{
		int num = NRestSiteRoom.Instance.Options.IndexOf(option);
		if (num < 0)
		{
			throw new InvalidOperationException($"Rest site option {option} was selected, but it was not in the list of rest site options!");
		}
		_executingOption = true;
		NRestSiteRoom.Instance.DisableOptions();
		RefreshTextState();
		bool success = false;
		try
		{
			success = await RunManager.Instance.RestSiteSynchronizer.ChooseLocalOption(num);
			if (success)
			{
				NRestSiteRoom.Instance.AfterSelectingOption(option);
			}
		}
		finally
		{
			_executingOption = false;
			if (this.IsValid())
			{
				RefreshTextState();
			}
			if (!success && this.IsValid())
			{
				await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
				NRestSiteRoom.Instance?.EnableOptions();
			}
		}
	}

	protected override void OnPress()
	{
		if (!_isUnclickable)
		{
			base.OnPress();
			_currentTween?.Kill();
			_currentTween = CreateTween().SetParallel();
			_currentTween.TweenProperty(_visuals, "scale", Vector2.One * 0.9f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_currentTween.TweenProperty(_label, "position", _labelPosition, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_currentTween.TweenProperty(_outline, "scale", Vector2.One * 0.9f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
			_currentTween.TweenMethod(Callable.From<float>(UpdateShaderParam), 1.2f, 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		_currentTween?.Kill();
		_currentTween = CreateTween().SetParallel();
		_currentTween.TweenProperty(_visuals, "scale", Vector2.One * 1.1f, 0.05);
		_currentTween.TweenProperty(_outline, "scale", Vector2.One, 0.05);
		_currentTween.TweenProperty(_label, "position", _labelPosition + new Vector2(0f, 6f), 0.05);
		if (!_isUnclickable)
		{
			_hsv.SetShaderParameter(_v, 1.2f);
		}
		RefreshTextState();
	}

	protected override void OnUnfocus()
	{
		_currentTween?.Kill();
		_currentTween = CreateTween().SetParallel();
		_currentTween.TweenProperty(_visuals, "scale", Vector2.One, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_currentTween.TweenProperty(_label, "position", _labelPosition, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		_currentTween.TweenProperty(_outline, "scale", Vector2.One * 0.9f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		if (!_isUnclickable)
		{
			_currentTween.TweenMethod(Callable.From<float>(UpdateShaderParam), 1.2f, 1f, 1.0).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
		}
		RefreshTextState();
	}

	public void RefreshTextState()
	{
		NRestSiteRoom instance = NRestSiteRoom.Instance;
		if (instance != null)
		{
			if (base.IsFocused || _executingOption)
			{
				instance.SetText(Option.Description.GetFormattedText());
			}
			else
			{
				instance.FadeOutOptionDescription();
			}
		}
	}

	private void UpdateShaderParam(float value)
	{
		_hsv.SetShaderParameter(_v, value);
	}
}
