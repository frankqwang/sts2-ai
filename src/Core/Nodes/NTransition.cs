using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes;

public partial class NTransition : ColorRect
{
	private static readonly StringName _threshold = new StringName("threshold");

	private const string _fightTransitionPath = "res://materials/transitions/fight_transition_mat.tres";

	private const string _fadeTransitionPath = "res://materials/transitions/fade_transition_mat.tres";

	private float _initialGradientYPosition;

	private float _targetGradientYPosition;

	private Control _gradientTransition;

	private Control _simpleTransition;

	private Tween? _tween;

	public bool InTransition { get; private set; }

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlyArray<string>(new string[2] { "res://materials/transitions/fight_transition_mat.tres", "res://materials/transitions/fade_transition_mat.tres" });

	public override void _Ready()
	{
		_gradientTransition = GetNode<Control>("GradientTransition");
		_simpleTransition = GetNode<Control>("SimpleTransition");
		_initialGradientYPosition = _gradientTransition.Position.Y;
		_targetGradientYPosition = 0f;
	}

	public async Task FadeOut(float time = 0.8f, string transitionPath = "res://materials/transitions/fade_transition_mat.tres", CancellationToken? cancelToken = null)
	{
		if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			InTransition = true;
			base.Visible = false;
			return;
		}
		InTransition = true;
		Control simpleTransition = _simpleTransition;
		Color modulate = _simpleTransition.Modulate;
		modulate.A = 0f;
		simpleTransition.Modulate = modulate;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_simpleTransition, "modulate:a", 1f, time).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Quad);
		if (!(base.Material is ShaderMaterial shaderMaterial))
		{
			Log.Warn("NTransition.Material is null or not a ShaderMaterial (actual: " + (base.Material?.GetType().Name ?? "null") + "). Skipping transition.");
		}
		else
		{
			if (shaderMaterial.GetShaderParameter(_threshold).AsInt32() == 1)
			{
				return;
			}
			base.Material = PreloadManager.Cache.GetMaterial(transitionPath);
			Material material = base.Material;
			if (!(material is ShaderMaterial transitionMaterial))
			{
				Log.Warn("NTransition.Material failed to load from cache (path: " + transitionPath + "). Skipping transition.");
				return;
			}
			transitionMaterial.SetShaderParameter(_threshold, 0);
			double t = 0.0;
			while (t < (double)time)
			{
				if (cancelToken.HasValue && cancelToken.GetValueOrDefault().IsCancellationRequested)
				{
					_tween?.FastForwardToCompletion();
					break;
				}
				transitionMaterial.SetShaderParameter(_threshold, 1.0 - ((double)time - t));
				t += GetProcessDeltaTime();
				await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			}
			base.MouseFilter = MouseFilterEnum.Stop;
			transitionMaterial.SetShaderParameter(_threshold, 1);
		}
	}

	public async Task FadeIn(float time = 0.8f, string transitionPath = "res://materials/transitions/fade_transition_mat.tres", CancellationToken? cancelToken = null)
	{
		if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			base.Visible = false;
			InTransition = false;
			base.MouseFilter = MouseFilterEnum.Ignore;
			return;
		}
		_tween?.Kill();
		Control simpleTransition = _simpleTransition;
		Color modulate = _simpleTransition.Modulate;
		modulate.A = 0f;
		simpleTransition.Modulate = modulate;
		if (!(base.Material is ShaderMaterial shaderMaterial))
		{
			Log.Warn("NTransition.Material is null or not a ShaderMaterial (actual: " + (base.Material?.GetType().Name ?? "null") + "). Skipping transition.");
			InTransition = false;
			return;
		}
		if (shaderMaterial.GetShaderParameter(_threshold).AsInt32() == 0)
		{
			InTransition = false;
			return;
		}
		base.Material = PreloadManager.Cache.GetMaterial(transitionPath);
		Material material = base.Material;
		if (!(material is ShaderMaterial transitionMaterial))
		{
			Log.Warn("NTransition.Material failed to load from cache (path: " + transitionPath + "). Skipping transition.");
			InTransition = false;
			return;
		}
		transitionMaterial.SetShaderParameter(_threshold, 1);
		base.MouseFilter = MouseFilterEnum.Ignore;
		double t = 0.0;
		while (t < (double)time)
		{
			if (cancelToken.HasValue && cancelToken.GetValueOrDefault().IsCancellationRequested)
			{
				_tween?.FastForwardToCompletion();
				break;
			}
			transitionMaterial.SetShaderParameter(_threshold, Ease.CubicIn((float)((double)time - t)));
			t += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			if (t / (double)time > 0.75)
			{
				InTransition = false;
			}
		}
		InTransition = false;
		transitionMaterial.SetShaderParameter(_threshold, 0);
		base.MouseFilter = MouseFilterEnum.Ignore;
	}

	public async Task RoomFadeOut()
	{
		InTransition = true;
		if (!TestMode.IsOn && SaveManager.Instance.PrefsSave.FastMode != FastModeType.Instant)
		{
			Control simpleTransition = _simpleTransition;
			Color modulate = _simpleTransition.Modulate;
			modulate.A = 0f;
			simpleTransition.Modulate = modulate;
			Control gradientTransition = _gradientTransition;
			modulate = _gradientTransition.Modulate;
			modulate.A = 1f;
			gradientTransition.Modulate = modulate;
			Control gradientTransition2 = _gradientTransition;
			Vector2 position = _gradientTransition.Position;
			position.Y = _initialGradientYPosition;
			gradientTransition2.Position = position;
			_tween?.Kill();
			_tween = CreateTween().SetParallel();
			if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Normal)
			{
				_tween.TweenProperty(_gradientTransition, "position:y", _targetGradientYPosition, 0.6).SetDelay(0.5);
				_tween.TweenProperty(_simpleTransition, "modulate:a", 1f, 0.6).SetDelay(0.5);
			}
			else if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast)
			{
				_tween.TweenProperty(_simpleTransition, "modulate:a", 1f, 0.3).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Quad)
					.SetDelay(0.3);
			}
			await ToSignal(_tween, Tween.SignalName.Finished);
		}
	}

	public async Task RoomFadeIn(bool showTransition = true)
	{
		if (TestMode.IsOn)
		{
			return;
		}
		Color modulate;
		if (!showTransition || SaveManager.Instance.PrefsSave.FastMode == FastModeType.Instant)
		{
			Control simpleTransition = _simpleTransition;
			modulate = _simpleTransition.Modulate;
			modulate.A = 0f;
			simpleTransition.Modulate = modulate;
		}
		if (!(base.Material is ShaderMaterial shaderMaterial))
		{
			Log.Warn("NTransition.Material is null or not a ShaderMaterial (actual: " + (base.Material?.GetType().Name ?? "null") + "). Skipping transition.");
			InTransition = false;
			return;
		}
		shaderMaterial.SetShaderParameter(_threshold, 0);
		Control gradientTransition = _gradientTransition;
		modulate = _gradientTransition.Modulate;
		modulate.A = 0f;
		gradientTransition.Modulate = modulate;
		Control simpleTransition2 = _simpleTransition;
		modulate = _simpleTransition.Modulate;
		modulate.A = 1f;
		simpleTransition2.Modulate = modulate;
		_tween?.Kill();
		_tween = CreateTween().SetParallel();
		if (SaveManager.Instance.PrefsSave.FastMode == FastModeType.Fast)
		{
			_tween.TweenProperty(_simpleTransition, "modulate:a", 0f, 0.3);
			base.MouseFilter = MouseFilterEnum.Ignore;
		}
		else
		{
			_tween.TweenProperty(_simpleTransition, "modulate:a", 0f, 0.8);
			_tween.TweenCallback(Callable.From(delegate
			{
				base.MouseFilter = MouseFilterEnum.Ignore;
				InTransition = false;
			})).SetDelay(0.2);
		}
		await ToSignal(_tween, Tween.SignalName.Finished);
		InTransition = false;
	}
}
