using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;

namespace MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;

public partial class NSpriteAnimator : Sprite2D
{
	[ExportGroup("Animation Settings", "")]
	[Export(PropertyHint.None, "")]
	private Texture2D[] _frames;

	[Export(PropertyHint.None, "")]
	private float _fps = 15f;

	[Export(PropertyHint.None, "")]
	private bool _loop;

	[ExportGroup("Rotation Settings", "")]
	[Export(PropertyHint.None, "")]
	private bool _randomizeRotation;

	[Export(PropertyHint.None, "")]
	private Vector2 _rotationRange;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	public override void _Ready()
	{
		if (_randomizeRotation)
		{
			base.RotationDegrees = new System.Random().Next((int)_rotationRange.X, (int)_rotationRange.Y);
		}
		TaskHelper.RunSafely(PlayAnimation());
	}

	public override void _ExitTree()
	{
		_cancelToken.Cancel();
	}

	private async Task PlayAnimation()
	{
		int i = 0;
		int interval = Mathf.RoundToInt(1000f / _fps);
		while (!_cancelToken.IsCancellationRequested)
		{
			base.Texture = _frames[i];
			i++;
			if (_loop)
			{
				i %= _frames.Length;
			}
			await Task.Delay(interval, _cancelToken.Token);
			if (_frames.Length <= i)
			{
				break;
			}
		}
		if (!_cancelToken.IsCancellationRequested)
		{
			this.QueueFreeSafely();
		}
	}
}
