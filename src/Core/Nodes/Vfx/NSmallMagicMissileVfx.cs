using System.Threading;
using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NSmallMagicMissileVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_small_magic_missile");

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _anticipationParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _projectileStartParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _projectileParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _impactParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Array<GpuParticles2D> _modulateParticles = new Array<GpuParticles2D>();

	[Export(PropertyHint.None, "")]
	private Node2D? _anticipationContainer;

	[Export(PropertyHint.None, "")]
	private float _anticipationDuration = 0.2f;

	[Export(PropertyHint.None, "")]
	private Node2D? _projectileContainer;

	[Export(PropertyHint.None, "")]
	private Node2D? _projectileStartPoint;

	[Export(PropertyHint.None, "")]
	private Node2D? _projectileEndPoint;

	[Export(PropertyHint.None, "")]
	private float _projectileOffset = 100f;

	private CancellationTokenSource? _cts;

	[field: Export(PropertyHint.None, "")]
	public float WaitTime { get; private set; } = 0.2f;

	public static NSmallMagicMissileVfx? Create(Vector2 targetCenterPosition, Color tint)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NSmallMagicMissileVfx nSmallMagicMissileVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NSmallMagicMissileVfx>(PackedScene.GenEditState.Disabled);
		nSmallMagicMissileVfx.GlobalPosition = targetCenterPosition;
		nSmallMagicMissileVfx.Initialize();
		nSmallMagicMissileVfx.ModulateParticles(tint);
		return nSmallMagicMissileVfx;
	}

	private Vector2 GetProjectileDirection()
	{
		Vector3 vector = Quaternion.FromEuler(new Vector3(0f, 0f, Mathf.DegToRad(-30f))) * Vector3.Up;
		return new Vector2(vector.X, vector.Y).Normalized();
	}

	private Vector2 GetTopPosition(Vector2 projectileDirection)
	{
		return (Vector2)Geometry2D.LineIntersectsLine(base.GlobalPosition, projectileDirection, new Vector2(0f, 80f), Vector2.Right);
	}

	private void Initialize()
	{
		Vector2 projectileDirection = GetProjectileDirection();
		Vector2 topPosition = GetTopPosition(projectileDirection);
		_anticipationContainer.GlobalPosition = topPosition;
		_projectileStartPoint.GlobalPosition = topPosition + projectileDirection * _projectileOffset;
		_projectileEndPoint.GlobalPosition = base.GlobalPosition + projectileDirection * _projectileOffset;
		_projectileContainer.Visible = false;
	}

	private void ModulateParticles(Color tint)
	{
		for (int i = 0; i < _modulateParticles.Count; i++)
		{
			_modulateParticles[i].SelfModulate = tint;
		}
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	public override void _ExitTree()
	{
		_cts?.Cancel();
		_cts?.Dispose();
	}

	private async Task PlaySequence()
	{
		_cts = new CancellationTokenSource();
		for (int i = 0; i < _anticipationParticles.Count; i++)
		{
			_anticipationParticles[i].Restart();
		}
		await Cmd.Wait(_anticipationDuration, _cts.Token);
		for (int j = 0; j < _projectileStartParticles.Count; j++)
		{
			_projectileStartParticles[j].Restart();
		}
		_projectileContainer.GlobalPosition = _projectileStartPoint.GlobalPosition;
		_projectileContainer.Visible = true;
		for (int k = 0; k < _projectileParticles.Count; k++)
		{
			_projectileParticles[k].Restart();
		}
		double timer = 0.0;
		while (timer < (double)WaitTime && !_cts.IsCancellationRequested)
		{
			float weight = (float)timer / WaitTime;
			_projectileContainer.GlobalPosition = _projectileStartPoint.GlobalPosition.Lerp(_projectileEndPoint.GlobalPosition, weight);
			timer += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		if (!_cts.IsCancellationRequested)
		{
			_projectileContainer.Visible = false;
			for (int l = 0; l < _impactParticles.Count; l++)
			{
				_impactParticles[l].Restart();
			}
			NGame.Instance?.ScreenShake(ShakeStrength.Medium, ShakeDuration.Short);
			await Cmd.Wait(2f, _cts.Token);
			this.QueueFreeSafely();
		}
	}
}
