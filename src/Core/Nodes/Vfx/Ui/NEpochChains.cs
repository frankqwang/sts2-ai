using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NEpochChains : TextureRect
{
	[Signal]
	public delegate void OnAnimationFinishedEventHandler();

	[Export(PropertyHint.None, "")]
	private float _duration = 0.5f;

	[Export(PropertyHint.None, "")]
	private Array<NParticlesContainer>? _particles;

	[Export(PropertyHint.None, "")]
	private NParticlesContainer? _endParticles;

	[Export(PropertyHint.None, "")]
	private Curve? _particlesCurve;

	[Export(PropertyHint.None, "")]
	private Curve _brightEnabledCurve;

	[Export(PropertyHint.None, "")]
	private Curve _erosionEnabledCurve;

	[Export(PropertyHint.None, "")]
	private Curve _erosionBaseCurve;

	private static readonly StringName _brightEnabledString = new StringName("bright_enabled");

	private static readonly StringName _erosionEnabledString = new StringName("erosion_enabled");

	private static readonly StringName _erosionBaseString = new StringName("erosion_base");

	private int _previousParticleIndex = -1;

	private ShaderMaterial? _asShaderMaterial;

	private void UpdateParticles(int index)
	{
		if (_previousParticleIndex == index)
		{
			return;
		}
		_previousParticleIndex = index;
		for (int i = 0; i < _particles.Count; i++)
		{
			if (i == index)
			{
				_particles[i].Restart();
			}
		}
	}

	private void SetProperties(float interpolation)
	{
		if (_asShaderMaterial != null)
		{
			float num = _brightEnabledCurve.Sample(interpolation);
			float num2 = _erosionEnabledCurve.Sample(interpolation);
			float num3 = _erosionBaseCurve.Sample(interpolation);
			_asShaderMaterial.SetShaderParameter(_brightEnabledString, num);
			_asShaderMaterial.SetShaderParameter(_erosionEnabledString, num2);
			_asShaderMaterial.SetShaderParameter(_erosionBaseString, num3);
		}
	}

	public void Unlock()
	{
		TaskHelper.RunSafely(Unlocking());
	}

	public async Task Unlocking()
	{
		_previousParticleIndex = -1;
		base.SelfModulate = Colors.White;
		double timer = 0.0;
		Material originalMaterial = base.Material;
		_asShaderMaterial = (ShaderMaterial)originalMaterial.Duplicate(deep: true);
		base.Material = _asShaderMaterial;
		SetProperties(0f);
		while (timer < (double)_duration)
		{
			float num = (float)timer / _duration;
			float s = _particlesCurve.Sample(num);
			SetProperties(num);
			UpdateParticles(Mathf.FloorToInt(s));
			timer += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		SetProperties(1f);
		base.Material = originalMaterial;
		_asShaderMaterial.Dispose();
		base.SelfModulate = new Color(1f, 1f, 1f, 0f);
		_endParticles.Restart();
		EmitSignal(SignalName.OnAnimationFinished);
	}
}
