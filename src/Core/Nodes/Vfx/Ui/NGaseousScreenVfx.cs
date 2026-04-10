using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Ui;

public partial class NGaseousScreenVfx : AspectRatioContainer
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/ui/vfx_gaseous_screen");

	[Export(PropertyHint.None, "")]
	private ColorRect _gfx;

	[Export(PropertyHint.None, "")]
	private float _duration = 1f;

	[Export(PropertyHint.None, "")]
	private Curve? _alphaMultiplierCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _minBaseAlphaCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _erosionCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _noiseAOffsetCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _noiseBOffsetCurve;

	private Material? _originalMaterial;

	private ShaderMaterial? _materialCopy;

	private float _noiseAOffsetY;

	private float _noiseBOffsetY;

	private static readonly StringName _alphaMultiplierString = new StringName("alpha_multiplier");

	private static readonly StringName _minBaseAlphaString = new StringName("min_base_alpha");

	private static readonly StringName _noiseAOffsetString = new StringName("noise_a_static_offset");

	private static readonly StringName _noiseBOffsetString = new StringName("noise_b_static_offset");

	private static readonly StringName _erosionString = new StringName("erosion_base");

	public static NGaseousScreenVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene(scenePath).Instantiate<NGaseousScreenVfx>(PackedScene.GenEditState.Disabled);
	}

	public override void _Ready()
	{
		_originalMaterial = _gfx.Material;
		_materialCopy = (ShaderMaterial)_originalMaterial.Duplicate(deep: true);
		_gfx.Material = _materialCopy;
		SetProperties(1f);
		Play();
	}

	private void SetProperties(float interpolation)
	{
		float num = _alphaMultiplierCurve.Sample(interpolation);
		float num2 = _minBaseAlphaCurve.Sample(interpolation);
		float x = _noiseAOffsetCurve.Sample(interpolation);
		float x2 = _noiseBOffsetCurve.Sample(interpolation);
		float num3 = _erosionCurve.Sample(interpolation);
		_materialCopy.SetShaderParameter(_alphaMultiplierString, num);
		_materialCopy.SetShaderParameter(_minBaseAlphaString, num2);
		_materialCopy.SetShaderParameter(_noiseAOffsetString, new Vector2(x, _noiseAOffsetY));
		_materialCopy.SetShaderParameter(_noiseBOffsetString, new Vector2(x2, _noiseBOffsetY));
		_materialCopy.SetShaderParameter(_erosionString, num3);
	}

	public void Play()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private async Task PlaySequence()
	{
		_noiseAOffsetY = GD.Randf();
		_noiseBOffsetY = GD.Randf();
		double timer = 0.0;
		SetProperties(0f);
		while (timer < (double)_duration)
		{
			float properties = (float)(timer / (double)_duration);
			SetProperties(properties);
			timer += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		SetProperties(1f);
		this.QueueFreeSafely();
	}
}
