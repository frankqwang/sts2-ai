using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NHorizontalLinesVfx : GpuParticles2D
{
	private Tween? _tween;

	private static readonly string _scenePath = SceneHelper.GetScenePath("vfx/whole_screen/horizontal_lines_vfx");

	private double _duration;

	private ParticleProcessMaterial _mat;

	private bool _isMovingRight;

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(_scenePath);

	public static NHorizontalLinesVfx? Create(Color color, double duration = 2.0, bool movingRightwards = true)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NHorizontalLinesVfx nHorizontalLinesVfx = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NHorizontalLinesVfx>(PackedScene.GenEditState.Disabled);
		nHorizontalLinesVfx._duration = Mathf.Max(1.0, duration);
		nHorizontalLinesVfx._mat = (ParticleProcessMaterial)nHorizontalLinesVfx.ProcessMaterial;
		nHorizontalLinesVfx._mat.Color = color;
		nHorizontalLinesVfx._isMovingRight = movingRightwards;
		return nHorizontalLinesVfx;
	}

	public override void _Ready()
	{
		Control parent = GetParent<Control>();
		_mat.EmissionShapeOffset = new Vector3(-500f, parent.Size.Y * 0.5f, 0f);
		_mat.EmissionShapeScale = new Vector3(200f, parent.Size.Y * 0.5f, 1f);
		if (!_isMovingRight)
		{
			base.RotationDegrees = 180f;
			base.Position = new Vector2(parent.Size.X, parent.Size.Y);
		}
		TaskHelper.RunSafely(PlayAnim());
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task PlayAnim()
	{
		base.Modulate = StsColors.transparentWhite;
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.45);
		_tween.Chain();
		_tween.TweenInterval(_duration - 0.9);
		_tween.Chain();
		_tween.TweenProperty(this, "modulate:a", 0f, 0.45);
		await ToSignal(_tween, Tween.SignalName.Finished);
	}
}
