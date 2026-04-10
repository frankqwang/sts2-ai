using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NMapNodeSelectVfx : Control
{
	private const string _path = "res://scenes/vfx/map_node_select_vfx.tscn";

	private static readonly string[] _textures = new string[1] { "res://images/vfx/brush_particle_2.png" };

	private double _lifeTimer;

	private readonly CancellationTokenSource _cancelToken = new CancellationTokenSource();

	private GpuParticles2D _particles;

	public static IEnumerable<string> AssetPaths => _textures.Append("res://scenes/vfx/map_node_select_vfx.tscn");

	public override void _Ready()
	{
		_particles = GetNode<GpuParticles2D>("Particles");
		_particles.Emitting = true;
		TaskHelper.RunSafely(Play());
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		_cancelToken.Cancel();
	}

	public static NMapNodeSelectVfx? Create(float scaleMultiplier)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NMapNodeSelectVfx nMapNodeSelectVfx = PreloadManager.Cache.GetScene("res://scenes/vfx/map_node_select_vfx.tscn").Instantiate<NMapNodeSelectVfx>(PackedScene.GenEditState.Disabled);
		nMapNodeSelectVfx.Scale = Vector2.One * scaleMultiplier;
		return nMapNodeSelectVfx;
	}

	private async Task Play()
	{
		await Task.Delay(1000, _cancelToken.Token);
		if (!_cancelToken.IsCancellationRequested)
		{
			this.QueueFreeSafely();
		}
	}
}
