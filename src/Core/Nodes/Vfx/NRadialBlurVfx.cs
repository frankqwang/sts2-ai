using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NRadialBlurVfx : BackBufferCopy
{
	private static readonly StringName _blurCenterx = new StringName("blur_center:x");

	private ShaderMaterial _blurShader;

	private VfxPosition _vfxPosition;

	private Control _rect;

	private Tween? _tween;

	public override void _Ready()
	{
		_rect = GetNode<Control>("Rect");
		_blurShader = (ShaderMaterial)_rect.GetMaterial();
	}

	public void Activate(VfxPosition vfxPosition = VfxPosition.Center)
	{
		if (!TestMode.IsOn && !base.Visible)
		{
			base.Visible = true;
			_rect.SetDeferred(Control.PropertyName.Size, GetViewportRect().Size);
			switch (vfxPosition)
			{
			case VfxPosition.Left:
				_blurShader.SetShaderParameter(_blurCenterx, 0.3f);
				break;
			case VfxPosition.Right:
				_blurShader.SetShaderParameter(_blurCenterx, 0.6f);
				break;
			default:
				_blurShader.SetShaderParameter(_blurCenterx, 0.45f);
				break;
			}
			TaskHelper.RunSafely(Animate());
		}
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	private async Task Animate()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(_blurShader, "shader_parameter/blur_power", 0.005f, 0.10000000149011612);
		_tween.Chain();
		_tween.TweenProperty(_blurShader, "shader_parameter/blur_power", 0f, 0.8999999761581421).SetEase(Tween.EaseType.In).SetTrans(Tween.TransitionType.Cubic);
		await ToSignal(_tween, Tween.SignalName.Finished);
		base.Visible = false;
	}
}
