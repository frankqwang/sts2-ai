using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Ui;

public partial class NLowHpBorderVfx : ColorRect
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/ui/vfx_low_hp_border");

	[Export(PropertyHint.None, "")]
	private float _duration = 1f;

	[Export(PropertyHint.None, "")]
	private Curve? _alphaMultiplierCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _noiseOffsetCurve;

	[Export(PropertyHint.None, "")]
	private Gradient? _gradient;

	private Material? _originalMaterial;

	private ShaderMaterial? _materialCopy;

	private bool _isPlaying;

	private double _currentTimer;

	private static readonly StringName _alphaMultiplierString = new StringName("alpha_multiplier");

	private static readonly StringName _noiseInitialOffsetString = new StringName("noise_initial_offset");

	private static readonly StringName _mainColorString = new StringName("main_color");

	public static NLowHpBorderVfx Create()
	{
		return PreloadManager.Cache.GetScene(scenePath).Instantiate<NLowHpBorderVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_isPlaying = false;
		_originalMaterial = base.Material;
		_materialCopy = (ShaderMaterial)_originalMaterial.Duplicate(deep: true);
		base.Material = _materialCopy;
		SetProperties(1f);
	}

	private void SetProperties(float interpolation)
	{
		float num = _alphaMultiplierCurve.Sample(interpolation);
		Color color = _gradient.Sample(interpolation);
		_materialCopy.SetShaderParameter(_alphaMultiplierString, num);
		_materialCopy.SetShaderParameter(_mainColorString, color);
	}

	private void RandomizeInitialOffset()
	{
		_materialCopy.SetShaderParameter(_noiseInitialOffsetString, new Vector2(GD.Randf(), GD.Randf()));
	}

	public void Play()
	{
		if (_isPlaying)
		{
			_currentTimer = 0.0;
		}
		else
		{
			TaskHelper.RunSafely(PlaySequence());
		}
	}

	private async Task PlaySequence()
	{
		_isPlaying = true;
		_currentTimer = 0.0;
		RandomizeInitialOffset();
		SetProperties(0f);
		while (_currentTimer < (double)_duration)
		{
			float properties = (float)(_currentTimer / (double)_duration);
			SetProperties(properties);
			_currentTimer += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		SetProperties(1f);
		_isPlaying = false;
	}
}
