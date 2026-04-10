using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NItemThrowVfx : Node2D
{
	public static readonly string scenePath = SceneHelper.GetScenePath("vfx/vfx_item_throw");

	private const float _baseItemSize = 80f;

	[Export(PropertyHint.None, "")]
	private Sprite2D? _itemSprite;

	[Export(PropertyHint.None, "")]
	private float _flightTime;

	[Export(PropertyHint.None, "")]
	private float _heightMultiplier;

	[Export(PropertyHint.None, "")]
	private Curve? _horizontalCurve;

	[Export(PropertyHint.None, "")]
	private Curve? _verticalCurve;

	[Export(PropertyHint.None, "")]
	private float _rotationMultiplier;

	[Export(PropertyHint.None, "")]
	private Curve? _rotationInfluenceCurve;

	private Vector2 _sourcePosition;

	private Vector2 _targetPosition;

	public static NItemThrowVfx? Create(Vector2 sourcePosition, Vector2 targetPosition, Texture2D? itemTexture, Vector2? scale = null)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NItemThrowVfx nItemThrowVfx = PreloadManager.Cache.GetScene(scenePath).Instantiate<NItemThrowVfx>(PackedScene.GenEditState.Disabled);
		nItemThrowVfx._sourcePosition = sourcePosition;
		nItemThrowVfx._targetPosition = targetPosition;
		if (nItemThrowVfx._itemSprite != null)
		{
			nItemThrowVfx._itemSprite.Visible = false;
			nItemThrowVfx._itemSprite.Scale = scale ?? Vector2.One;
			if (itemTexture != null)
			{
				nItemThrowVfx._itemSprite.Texture = itemTexture;
				nItemThrowVfx._itemSprite.Scale *= 80f / (float)itemTexture.GetWidth();
			}
		}
		return nItemThrowVfx;
	}

	public override void _Ready()
	{
		TaskHelper.RunSafely(ThrowItem());
	}

	private async Task ThrowItem()
	{
		_itemSprite.Visible = true;
		_itemSprite.GlobalPosition = _sourcePosition;
		_itemSprite.RotationDegrees = Rng.Chaotic.NextFloat(360f);
		double timer = 0.0;
		while (timer < (double)_flightTime)
		{
			double processDeltaTime = GetProcessDeltaTime();
			float offset = (float)(timer / (double)_flightTime);
			float weight = _horizontalCurve.Sample(offset);
			float num = _verticalCurve.Sample(offset);
			float num2 = _rotationInfluenceCurve.Sample(offset);
			_itemSprite.Rotate((float)Mathf.DegToRad((double)(num2 * _rotationMultiplier) * processDeltaTime));
			Vector2 globalPosition = _sourcePosition.Lerp(_targetPosition, weight) + Vector2.Up * num * _heightMultiplier;
			_itemSprite.GlobalPosition = globalPosition;
			timer += processDeltaTime;
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
		}
		_itemSprite.Visible = false;
		this.QueueFreeSafely();
	}
}
