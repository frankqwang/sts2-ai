using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NLiquidOverlayVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_liquid_overlay");

	private Color _tint = Colors.White;

	private Creature? _targetCreature;

	private CancellationTokenSource? _cts;

	public static NLiquidOverlayVfx? Create(Creature? targetCreature, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NLiquidOverlayVfx nLiquidOverlayVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NLiquidOverlayVfx>(PackedScene.GenEditState.Disabled);
		nLiquidOverlayVfx._targetCreature = targetCreature;
		nLiquidOverlayVfx._tint = tint;
		return nLiquidOverlayVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlayVfx());
	}

	public override void _ExitTree()
	{
		_cts?.Cancel();
		_cts?.Dispose();
	}

	private async Task PlayVfx()
	{
		_cts = new CancellationTokenSource();
		if (_targetCreature != null)
		{
			(NCombatRoom.Instance?.GetCreatureNode(_targetCreature))?.Visuals.TryApplyLiquidOverlay(_tint);
		}
		await Cmd.Wait(0.5f, _cts.Token);
		this.QueueFreeSafely();
	}
}
