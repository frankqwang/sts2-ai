using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NDecimillipedeRocksVfx : Node2D
{
	[Export(PropertyHint.None, "")]
	private Node2D[] _rocks;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	public override void _Ready()
	{
		TaskHelper.RunSafely(Play(_cancelToken.Token));
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_cancelToken.Cancel();
	}

	private async Task Play(CancellationToken cancellationToken)
	{
		Node2D[] rocks = _rocks;
		foreach (Node2D rock in rocks)
		{
			await Task.Delay(Rng.Chaotic.NextInt(100, 201), cancellationToken);
			new MegaSprite(rock).GetAnimationState().SetAnimation($"fall{Rng.Chaotic.NextInt(1, 5)}", loop: false);
		}
		await Task.Delay(5000, cancellationToken);
		if (!cancellationToken.IsCancellationRequested)
		{
			this.QueueFreeSafely();
		}
	}
}
