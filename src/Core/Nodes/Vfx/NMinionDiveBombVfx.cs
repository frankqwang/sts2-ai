using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Rooms;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NMinionDiveBombVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_minion_dive_bomb");

	[Export(PropertyHint.None, "")]
	private Sprite2D? _minionSprite;

	[Export(PropertyHint.None, "")]
	private Array<Texture2D>? _minionTextures;

	[Export(PropertyHint.None, "")]
	private AnimationPlayer? _minionAnimator;

	[Export(PropertyHint.None, "")]
	private Array<string>? _minionAnimations;

	[Export(PropertyHint.None, "")]
	private Array<NParticlesContainer>? _minionVfx;

	[Export(PropertyHint.None, "")]
	private Node2D? _fallingTrail;

	[Export(PropertyHint.None, "")]
	private NParticlesContainer? _fallingVfx;

	[Export(PropertyHint.None, "")]
	private NParticlesContainer? _impactVfx;

	[Export(PropertyHint.None, "")]
	private float _flightTime;

	[Export(PropertyHint.None, "")]
	private float _fallingVfxEntryTime;

	[Export(PropertyHint.None, "")]
	private Curve? _horizontalCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _verticalCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _textureCurve;

	[Export(PropertyHint.None, "")]
	private float _maxHeight;

	[Export(PropertyHint.None, "")]
	private Vector2 _sourceOffset;

	[Export(PropertyHint.None, "")]
	private Vector2 _destinationOffset;

	private int _previousIndex = -1;

	private Vector2 _sourcePosition;

	private Vector2 _destinationPosition;

	private Vector2 SourceFinalPosition => _sourcePosition + _sourceOffset;

	private Vector2 DestinationFinalPosition => _destinationPosition + _destinationOffset;

	public static NMinionDiveBombVfx? Create(Creature owner, Creature target)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NCreature nCreature = NCombatRoom.Instance?.GetCreatureNode(owner);
		NCreature nCreature2 = NCombatRoom.Instance?.GetCreatureNode(target);
		if (nCreature2 != null && nCreature != null)
		{
			return Create(nCreature.VfxSpawnPosition, nCreature2.GetBottomOfHitbox());
		}
		return null;
	}

	public static NMinionDiveBombVfx? Create(Vector2 playerCenterPosition, Vector2 targetFloorPosition)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NMinionDiveBombVfx nMinionDiveBombVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NMinionDiveBombVfx>(PackedScene.GenEditState.Disabled);
		nMinionDiveBombVfx.Initialize(playerCenterPosition, targetFloorPosition);
		return nMinionDiveBombVfx;
	}

	private void Initialize(Vector2 sourcePosition, Vector2 destinationPosition)
	{
		_sourcePosition = sourcePosition;
		_destinationPosition = destinationPosition;
		_fallingTrail.Visible = true;
		base.GlobalPosition = sourcePosition;
		for (int i = 0; i < _minionVfx.Count; i++)
		{
			_minionVfx[i].SetEmitting(emitting: false);
		}
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(PlaySequence());
	}

	private void SetMinionVisible(bool visible)
	{
		_minionSprite.SelfModulate = (visible ? new Color(1f, 1f, 1f) : new Color(1f, 1f, 1f, 0f));
	}

	private void UpdateMinionSprite(int index)
	{
		if (_previousIndex == index)
		{
			return;
		}
		_previousIndex = index;
		Texture2D texture = _minionTextures[Mathf.Clamp(index, 0, _minionTextures.Count - 1)];
		_minionSprite.Texture = texture;
		string text = _minionAnimations[Mathf.Clamp(index, 0, _minionAnimations.Count - 1)];
		if (!_minionAnimator.CurrentAnimation.Equals(text))
		{
			_minionAnimator.Play(text);
		}
		for (int i = 0; i < _minionVfx.Count; i++)
		{
			if (i == index)
			{
				_minionVfx[i].Restart();
			}
		}
	}

	private async Task PlaySequence()
	{
		Vector2 startPos = SourceFinalPosition;
		Vector2 endPos = DestinationFinalPosition;
		UpdateMinionSprite(0);
		_minionSprite.GlobalPosition = startPos;
		SetMinionVisible(visible: true);
		double timer = 0.0;
		bool isPlayingFallingVfx = false;
		while (timer < (double)_flightTime)
		{
			float offset = (float)timer / _flightTime;
			float weight = _horizontalCurve.Sample(offset);
			float num = _verticalCurve.Sample(offset);
			float s = _textureCurve.Sample(offset);
			UpdateMinionSprite(Mathf.FloorToInt(s));
			Vector2 globalPosition = startPos.Lerp(endPos, weight);
			globalPosition += Vector2.Up * num * _maxHeight;
			_minionSprite.GlobalPosition = globalPosition;
			_fallingVfx.GlobalPosition = globalPosition;
			if (timer >= (double)_fallingVfxEntryTime && !isPlayingFallingVfx)
			{
				_fallingVfx.Restart();
				_fallingTrail.Visible = true;
				isPlayingFallingVfx = true;
			}
			timer += GetProcessDeltaTime();
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		SetMinionVisible(visible: false);
		NGame.Instance?.ScreenShake(ShakeStrength.Weak, ShakeDuration.Short);
		_impactVfx.GlobalPosition = _destinationPosition;
		_impactVfx.Restart();
		_fallingTrail.Visible = false;
		_fallingVfx.SetEmitting(emitting: false);
		await Cmd.Wait(2f);
		this.QueueFreeSafely();
	}
}
