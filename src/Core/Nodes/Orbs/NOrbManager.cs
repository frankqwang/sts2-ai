using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.Orbs;

public partial class NOrbManager : Control
{
	private Control _orbContainer;

	private readonly List<NOrb> _orbs = new List<NOrb>();

	private NCreature _creatureNode;

	private const float _minRadius = 225f;

	private const float _maxRadius = 300f;

	private const float _range = 150f;

	private const float _angleOffset = -25f;

	private const float _tweenSpeed = 0.45f;

	private Tween? _curTween;

	private static string ScenePath => SceneHelper.GetScenePath("/orbs/orb_manager");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	public bool IsLocal { get; private set; }

	private Player Player => _creatureNode.Entity.Player;

	public Control DefaultFocusOwner
	{
		get
		{
			if (_orbs.Count <= 0)
			{
				return _creatureNode.Hitbox;
			}
			return _orbs.First();
		}
	}

	public override void _Ready()
	{
		_orbContainer = GetNode<Control>("%Orbs");
	}

	public override void _EnterTree()
	{
		base._EnterTree();
		CombatManager.Instance.StateTracker.CombatStateChanged += OnCombatStateChanged;
		CombatManager.Instance.CombatSetUp += OnCombatSetup;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		CombatManager.Instance.StateTracker.CombatStateChanged -= OnCombatStateChanged;
		CombatManager.Instance.CombatSetUp -= OnCombatSetup;
	}

	public static NOrbManager Create(NCreature creature, bool isLocal)
	{
		if (creature.Entity.Player == null)
		{
			throw new InvalidOperationException("NOrbManager can only be applied to player creatures");
		}
		NOrbManager nOrbManager = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NOrbManager>(PackedScene.GenEditState.Disabled);
		nOrbManager._creatureNode = creature;
		nOrbManager.IsLocal = isLocal;
		return nOrbManager;
	}

	private void OnCombatSetup(CombatState _)
	{
		if (Player.Creature.IsAlive && Player.PlayerCombatState != null)
		{
			AddSlotAnim(Player.PlayerCombatState.OrbQueue.Capacity);
		}
	}

	public void RemoveSlotAnim(int amount)
	{
		if (amount > _orbs.Count)
		{
			throw new InvalidOperationException("There are not enough slots to remove.");
		}
		for (int i = 0; i < amount; i++)
		{
			NOrb nOrb = _orbs.Last();
			nOrb.QueueFreeSafely();
			_orbs.Remove(nOrb);
			if (nOrb.HasFocus())
			{
				_creatureNode.Hitbox.TryGrabFocus();
			}
		}
		TweenLayout();
		UpdateControllerNavigation();
	}

	public void AddSlotAnim(int amount)
	{
		for (int i = 0; i < amount; i++)
		{
			NOrb nOrb = NOrb.Create(LocalContext.IsMe(Player));
			_orbContainer.AddChildSafely(nOrb);
			_orbs.Add(nOrb);
			nOrb.Position = Vector2.Zero;
		}
		TweenLayout();
		UpdateControllerNavigation();
	}

	public void ReplaceOrb(OrbModel oldOrb, OrbModel newOrb)
	{
		for (int i = 0; i < _orbs.Count; i++)
		{
			if (_orbs[i].Model == oldOrb)
			{
				_orbs[i].ReplaceOrb(newOrb);
			}
		}
		UpdateControllerNavigation();
	}

	public void AddOrbAnim()
	{
		OrbModel model = Player.PlayerCombatState.OrbQueue.Orbs.LastOrDefault();
		NOrb nOrb = _orbs.FirstOrDefault((NOrb node) => node.Model == null);
		if (nOrb == null)
		{
			EvokeOrbAnim(_orbs.First((NOrb node) => node.Model != null).Model);
			nOrb = (NOrb)_orbContainer.GetChildren().First((Node node) => ((NOrb)node).Model == null);
		}
		NOrb nOrb2 = NOrb.Create(LocalContext.IsMe(Player), model);
		nOrb.AddSibling(nOrb2);
		_orbs.Insert(_orbs.IndexOf(nOrb), nOrb2);
		nOrb2.Position = nOrb.Position;
		_orbContainer.RemoveChildSafely(nOrb);
		_orbs.Remove(nOrb);
		nOrb.QueueFreeSafely();
		TweenLayout();
		UpdateControllerNavigation();
	}

	public void EvokeOrbAnim(OrbModel orb)
	{
		NOrb nOrb = _orbs.Last((NOrb node) => node.Model == orb);
		Tween tween = CreateTween();
		_orbs.Remove(nOrb);
		tween.TweenProperty(nOrb, "modulate:a", 0, 0.25);
		tween.Chain().TweenCallback(Callable.From(nOrb.QueueFreeSafely));
		NOrb nOrb2 = NOrb.Create(LocalContext.IsMe(Player));
		_orbContainer.AddChildSafely(nOrb2);
		_orbs.Add(nOrb2);
		nOrb2.Position = Vector2.Zero;
		if (nOrb.HasFocus())
		{
			_creatureNode.Hitbox.TryGrabFocus();
		}
		TweenLayout();
		UpdateControllerNavigation();
	}

	private void UpdateControllerNavigation()
	{
		for (int i = 0; i < _orbs.Count; i++)
		{
			NOrb nOrb = _orbs[i];
			NodePath path;
			if (i <= 0)
			{
				List<NOrb> orbs = _orbs;
				path = orbs[orbs.Count - 1].GetPath();
			}
			else
			{
				path = _orbs[i - 1].GetPath();
			}
			nOrb.FocusNeighborRight = path;
			_orbs[i].FocusNeighborLeft = ((i < _orbs.Count - 1) ? _orbs[i + 1].GetPath() : _orbs[0].GetPath());
			_orbs[i].FocusNeighborTop = _orbs[i].GetPath();
			_orbs[i].FocusNeighborBottom = _creatureNode.Hitbox.GetPath();
		}
		_creatureNode.UpdateNavigation();
	}

	private void TweenLayout()
	{
		int capacity = Player.PlayerCombatState.OrbQueue.Capacity;
		if (capacity != 0)
		{
			float num = 125f;
			float num2 = num / (float)(capacity - 1);
			float num3 = Mathf.Lerp(225f, 300f, ((float)capacity - 3f) / 7f);
			if (!IsLocal)
			{
				num3 *= 0.75f;
			}
			_curTween?.Kill();
			_curTween = CreateTween().SetParallel();
			for (int i = 0; i < capacity; i++)
			{
				float s = float.DegreesToRadians(-25f - num);
				Vector2 vector = new Vector2(0f - Mathf.Cos(s), Mathf.Sin(s)) * num3;
				_curTween.TweenProperty(_orbs[i], "position", vector, 0.44999998807907104).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Sine);
				num -= num2;
			}
		}
	}

	private void OnCombatStateChanged(CombatState _)
	{
		UpdateVisuals(OrbEvokeType.None);
	}

	public void UpdateVisuals(OrbEvokeType evokeType)
	{
		foreach (NOrb orb in _orbs)
		{
			orb.UpdateVisuals(isEvoking: false);
		}
		switch (evokeType)
		{
		case OrbEvokeType.Front:
			_orbs.FirstOrDefault()?.UpdateVisuals(isEvoking: true);
			break;
		case OrbEvokeType.All:
		{
			foreach (NOrb orb2 in _orbs)
			{
				orb2.UpdateVisuals(isEvoking: true);
			}
			break;
		}
		case OrbEvokeType.None:
			break;
		}
	}

	public void ClearOrbs()
	{
		_curTween?.Kill();
		if (_orbs.Count == 0)
		{
			return;
		}
		_curTween = CreateTween();
		foreach (NOrb orb in _orbs)
		{
			_curTween.Parallel().TweenProperty(orb, "position", Vector2.Zero, 1.0).SetEase(Tween.EaseType.InOut)
				.SetTrans(Tween.TransitionType.Sine);
			_curTween.Parallel().TweenProperty(orb, "modulate:a", 0, 0.25);
		}
		foreach (NOrb orb2 in _orbs)
		{
			_curTween.Chain().TweenCallback(Callable.From(orb2.QueueFreeSafely));
		}
		_orbs.Clear();
	}
}
