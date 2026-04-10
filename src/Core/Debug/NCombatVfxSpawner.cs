using System.Threading.Tasks;
using Godot;
using Godot.Collections;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Ui;

namespace MegaCrit.Sts2.Core.Debug;

public partial class NCombatVfxSpawner : Control
{
	[Export(PropertyHint.None, "")]
	private Node2D _backCombatVfxContainer;

	[Export(PropertyHint.None, "")]
	private Control _combatVfxContainer;

	private WorldEnvironment _env;

	[Export(PropertyHint.None, "")]
	private Node2D _playerPosition;

	[Export(PropertyHint.None, "")]
	private Node2D _playerGroundPosition;

	[Export(PropertyHint.None, "")]
	private Node2D _enemyPosition;

	[Export(PropertyHint.None, "")]
	private Node2D _enemyGroundPosition;

	[Export(PropertyHint.None, "")]
	private Node2D _defectEyePosition;

	[Export(PropertyHint.None, "")]
	private NLowHpBorderVfx _lowHpBorderVfx;

	[Export(PropertyHint.None, "")]
	private NGaseousScreenVfx _gaseousScreenVfx;

	private decimal _damage = 10m;

	private bool _shiftPressed;

	public override void _Ready()
	{
	}

	public override void _Process(double delta)
	{
		base._Process(delta);
		_shiftPressed = Input.IsKeyPressed(Key.Shift);
	}

	public override void _Input(InputEvent inputEvent)
	{
		if (inputEvent is InputEventKey inputEventKey && inputEventKey.Keycode == Key.Q && inputEventKey.Pressed)
		{
			TestFunctionA(_shiftPressed);
		}
		if (inputEvent is InputEventKey inputEventKey2 && inputEventKey2.Keycode == Key.W && inputEventKey2.Pressed)
		{
			TestFunctionB(_shiftPressed);
		}
		if (inputEvent is InputEventKey inputEventKey3 && inputEventKey3.Keycode == Key.E && inputEventKey3.Pressed)
		{
			TestFunctionC(_shiftPressed);
		}
	}

	private static Color GetRandomColor()
	{
		return new Color(Mathf.Lerp(0.35f, 1f, GD.Randf()), Mathf.Lerp(0.35f, 1f, GD.Randf()), Mathf.Lerp(0.35f, 1f, GD.Randf()));
	}

	private void TestFunctionA(bool shiftPressed)
	{
		NSweepingBeamVfx child = NSweepingBeamVfx.Create(_defectEyePosition.GlobalPosition, new Array<Vector2> { _enemyPosition.GlobalPosition });
		_combatVfxContainer.AddChildSafely(child);
	}

	private void TestFunctionB(bool shiftPressed)
	{
		_gaseousScreenVfx.Play();
	}

	private void TestFunctionC(bool shiftPressed)
	{
		NWormyImpactVfx child = NWormyImpactVfx.Create(_playerGroundPosition.GlobalPosition, _playerPosition.GlobalPosition);
		_combatVfxContainer.AddChildSafely(child);
	}

	private void SpawnVfx()
	{
		NGaseousImpactVfx child = NGaseousImpactVfx.Create(_enemyPosition.GlobalPosition, GetRandomColor());
		_combatVfxContainer.AddChildSafely(child);
	}

	private async Task SpawningVfx()
	{
		NItemThrowVfx child = NItemThrowVfx.Create(_playerPosition.GlobalPosition + Vector2.Up * 150f, _enemyPosition.GlobalPosition, null);
		_combatVfxContainer.AddChildSafely(child);
		await Cmd.Wait(0.55f);
		NSplashVfx child2 = NSplashVfx.Create(_enemyPosition.GlobalPosition, new Color(0.25f, 1f, 0.4f));
		_combatVfxContainer.AddChildSafely(child2);
	}

	private async Task Hyperbeaming()
	{
		NHyperbeamVfx child = NHyperbeamVfx.Create(_defectEyePosition.GlobalPosition, _enemyPosition.GlobalPosition);
		_combatVfxContainer.AddChildSafely(child);
		await Cmd.Wait(NHyperbeamVfx.hyperbeamAnticipationDuration);
		NHyperbeamImpactVfx child2 = NHyperbeamImpactVfx.Create(_defectEyePosition.GlobalPosition, _enemyPosition.GlobalPosition);
		_combatVfxContainer.AddChildSafely(child2);
	}
}
