using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Screens.Shops;

[GlobalClass]
public partial class NMerchantHand : Node
{
	private Vector2 _startPos;

	private Vector2 _targetPos;

	private MegaBone _bone;

	private FastNoiseLite _noise;

	private float _time;

	private CancellationTokenSource? _stopPointingToken;

	private Control _rug;

	private Node2D _parent;

	private MegaSprite _animController;

	public override void _Ready()
	{
		_parent = GetParent<Node2D>();
		_animController = new MegaSprite(_parent);
		_rug = _parent.GetParent<Control>();
		_startPos = _parent.GlobalPosition;
		_targetPos = _startPos;
		_noise = new FastNoiseLite();
		_noise.NoiseType = FastNoiseLite.NoiseTypeEnum.Perlin;
		_noise.Frequency = 1f;
		_bone = _animController.GetSkeleton().FindBone("rotate_me");
		_animController.GetAnimationState().SetAnimation("default");
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_stopPointingToken?.Cancel();
	}

	public override void _Process(double delta)
	{
		_time += (float)delta;
		float x = _noise.GetNoise1D(_time * 0.1f) + 0.4f;
		float y = _noise.GetNoise1D((_time + 0.25f) * 0.1f) - 0.5f;
		_parent.GlobalPosition = _parent.GlobalPosition.Lerp(_targetPos + new Vector2(x, y) * 100f, (float)delta * 4f);
		_bone.SetRotation(Mathf.Lerp(-10f, 10f, (_parent.Position.X - _rug.Size.X * 0.5f - 50f) * 0.01f));
	}

	public void PointAtTarget(Vector2 pos)
	{
		_stopPointingToken?.Cancel();
		_targetPos = pos - Vector2.One * 50f;
	}

	public void StopPointing(float lingerTime)
	{
		_stopPointingToken?.Cancel();
		_stopPointingToken = new CancellationTokenSource();
		TaskHelper.RunSafely(WaitAndReturn(_stopPointingToken, lingerTime));
	}

	private async Task WaitAndReturn(CancellationTokenSource cancelToken, float lingerTime)
	{
		for (float timer = 0f; timer < lingerTime; timer += (float)GetProcessDeltaTime())
		{
			if (cancelToken.IsCancellationRequested || !this.IsValid() || !IsInsideTree())
			{
				return;
			}
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		_targetPos = _startPos;
	}
}
