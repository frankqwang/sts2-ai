using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Bindings.MegaSpine;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes.Combat;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Nodes.Vfx.Utilities;
using MegaCrit.Sts2.Core.Random;

namespace MegaCrit.Sts2.Core.Nodes.RestSite;

public partial class NRestSiteCharacter : Node2D
{
	private static readonly StringName _v = new StringName("v");

	private static readonly StringName _s = new StringName("s");

	private static readonly StringName _noise2Panning = new StringName("Noise2Panning");

	private static readonly StringName _noise1Panning = new StringName("Noise1Panning");

	private static readonly StringName _globalOffset = new StringName("GlobalOffset");

	private static readonly Vector2 _multiplayerConfirmationOffset = new Vector2(-25f, -123f);

	private static readonly Vector2 _multiplayerConfirmationFlipOffset = new Vector2(-155f, 0f);

	private static readonly string _multiplayerConfirmationScenePath = SceneHelper.GetScenePath("rest_site/rest_site_multiplayer_confirmation");

	private Control _controlRoot;

	private NSelectionReticle _selectionReticle;

	private Control _leftThoughtAnchor;

	private Control _rightThoughtAnchor;

	private int _characterIndex;

	private NThoughtBubbleVfx? _thoughtBubbleVfx;

	private CancellationTokenSource? _thoughtBubbleGoAwayCancellation;

	private Control? _selectedOptionConfirmation;

	private RestSiteOption? _hoveredRestSiteOption;

	private RestSiteOption? _selectingRestSiteOption;

	private RestSiteOption? _restSiteOptionInThoughtBubble;

	public Control Hitbox { get; private set; }

	public Player Player { get; private set; }

	public static NRestSiteCharacter Create(Player player, int characterIndex)
	{
		NRestSiteCharacter nRestSiteCharacter = PreloadManager.Cache.GetScene(player.Character.RestSiteAnimPath).Instantiate<NRestSiteCharacter>(PackedScene.GenEditState.Disabled);
		nRestSiteCharacter.Player = player;
		nRestSiteCharacter._characterIndex = characterIndex;
		return nRestSiteCharacter;
	}

	public override void _Ready()
	{
		_controlRoot = GetNode<Control>("ControlRoot");
		Hitbox = GetNode<Control>("%Hitbox");
		_selectionReticle = GetNode<NSelectionReticle>("%SelectionReticle");
		_leftThoughtAnchor = GetNode<Control>("%ThoughtBubbleLeft");
		_rightThoughtAnchor = GetNode<Control>("%ThoughtBubbleRight");
		string animationName = Player.RunState.CurrentActIndex switch
		{
			0 => "overgrowth_loop", 
			1 => "hive_loop", 
			2 => "glory_loop", 
			_ => throw new InvalidOperationException("Unexpected act"), 
		};
		foreach (Node2D childSpineNode in GetChildSpineNodes())
		{
			MegaTrackEntry megaTrackEntry = new MegaSprite(childSpineNode).GetAnimationState().SetAnimation(animationName);
			megaTrackEntry?.SetTrackTime(megaTrackEntry.GetAnimationEnd() * Rng.Chaotic.NextFloat());
		}
		if (Player.Character is Necrobinder)
		{
			Sprite2D node = GetNode<Sprite2D>("%NecroFire");
			Sprite2D node2 = GetNode<Sprite2D>("%OstyFire");
			RandomizeFire((ShaderMaterial)node.Material);
			RandomizeFire((ShaderMaterial)node2.Material);
			if (_characterIndex >= 2)
			{
				Node2D node3 = GetNode<Node2D>("Osty");
				Node2D node4 = GetNode<Node2D>("OstyRightAnchor");
				node3.Position = node4.Position;
				MoveChild(node3, 0);
			}
		}
		Hitbox.Connect(Control.SignalName.FocusEntered, Callable.From(OnFocus));
		Hitbox.Connect(Control.SignalName.FocusExited, Callable.From(OnUnfocus));
		Hitbox.Connect(Control.SignalName.MouseEntered, Callable.From(OnFocus));
		Hitbox.Connect(Control.SignalName.MouseExited, Callable.From(OnUnfocus));
	}

	private void RandomizeFire(ShaderMaterial mat)
	{
		mat.SetShaderParameter(_globalOffset, new Vector2(Rng.Chaotic.NextFloat(-50f, 50f), Rng.Chaotic.NextFloat(-50f, 50f)));
		mat.SetShaderParameter(_noise1Panning, mat.GetShaderParameter(_noise1Panning).AsVector2() + new Vector2(Rng.Chaotic.NextFloat(-0.1f, 0.1f), Rng.Chaotic.NextFloat(-0.1f, 0.1f)));
		mat.SetShaderParameter(_noise2Panning, mat.GetShaderParameter(_noise2Panning).AsVector2() + new Vector2(Rng.Chaotic.NextFloat(-0.1f, 0.1f), Rng.Chaotic.NextFloat(-0.1f, 0.1f)));
	}

	private void OnFocus()
	{
		if (NTargetManager.Instance.IsInSelection && NTargetManager.Instance.AllowedToTargetNode(this))
		{
			NTargetManager.Instance.OnNodeHovered(this);
			_selectionReticle.OnSelect();
			NRun.Instance.GlobalUi.MultiplayerPlayerContainer.HighlightPlayer(Player);
		}
	}

	private void OnUnfocus()
	{
		if (NTargetManager.Instance.IsInSelection && NTargetManager.Instance.AllowedToTargetNode(this))
		{
			NTargetManager.Instance.OnNodeUnhovered(this);
		}
		Deselect();
	}

	public void Deselect()
	{
		if (_selectionReticle.IsSelected)
		{
			_selectionReticle.OnDeselect();
		}
		NRun.Instance.GlobalUi.MultiplayerPlayerContainer.UnhighlightPlayer(Player);
	}

	public void FlipX()
	{
		Vector2 scale;
		foreach (Node2D childSpineNode in GetChildSpineNodes())
		{
			scale = childSpineNode.Scale;
			scale.X = 0f - childSpineNode.Scale.X;
			childSpineNode.Scale = scale;
			scale = childSpineNode.Position;
			scale.X = 0f - childSpineNode.Position.X;
			childSpineNode.Position = scale;
		}
		Control controlRoot = _controlRoot;
		scale = _controlRoot.Scale;
		scale.X = 0f - _controlRoot.Scale.X;
		controlRoot.Scale = scale;
	}

	public void HideFlameGlow()
	{
		foreach (Node2D childSpineNode in GetChildSpineNodes())
		{
			MegaSprite megaSprite = new MegaSprite(childSpineNode);
			if (megaSprite.HasAnimation("_tracks/light_off"))
			{
				megaSprite.GetAnimationState().SetAnimation("_tracks/light_off", loop: true, 1);
			}
		}
	}

	public void ShowHoveredRestSiteOption(RestSiteOption? option)
	{
		_hoveredRestSiteOption = option;
		RefreshThoughtBubbleVfx();
	}

	public void SetSelectingRestSiteOption(RestSiteOption? option)
	{
		_selectingRestSiteOption = option;
		RefreshThoughtBubbleVfx();
	}

	private void RefreshThoughtBubbleVfx()
	{
		if (_selectedOptionConfirmation != null)
		{
			return;
		}
		RestSiteOption restSiteOption = _selectingRestSiteOption ?? _hoveredRestSiteOption;
		if (_restSiteOptionInThoughtBubble == restSiteOption)
		{
			return;
		}
		_restSiteOptionInThoughtBubble = restSiteOption;
		if (restSiteOption == null)
		{
			TaskHelper.RunSafely(RemoveThoughtBubbleAfterDelay());
			return;
		}
		_thoughtBubbleGoAwayCancellation?.Cancel();
		if (_thoughtBubbleVfx == null)
		{
			int characterIndex = _characterIndex;
			bool flag = ((characterIndex == 0 || characterIndex == 3) ? true : false);
			bool flag2 = flag;
			_thoughtBubbleVfx = NThoughtBubbleVfx.Create(restSiteOption.Icon, (!flag2) ? DialogueSide.Left : DialogueSide.Right, null);
			ShaderMaterial shaderMaterial = (ShaderMaterial)_thoughtBubbleVfx.GetNode<TextureRect>("%Image").Material;
			shaderMaterial.SetShaderParameter(_s, 0.145f);
			shaderMaterial.SetShaderParameter(_v, 0.85f);
			this.AddChildSafely(_thoughtBubbleVfx);
			_thoughtBubbleVfx.GlobalPosition = GetRestSiteOptionAnchor().GlobalPosition;
		}
		else
		{
			_thoughtBubbleVfx.SetTexture(restSiteOption.Icon);
		}
	}

	public void ShowSelectedRestSiteOption(RestSiteOption option)
	{
		_thoughtBubbleVfx?.GoAway();
		_thoughtBubbleVfx = null;
		_selectedOptionConfirmation = PreloadManager.Cache.GetScene(_multiplayerConfirmationScenePath).Instantiate<Control>(PackedScene.GenEditState.Disabled);
		_selectedOptionConfirmation.GetNode<TextureRect>("%Icon").Texture = option.Icon;
		this.AddChildSafely(_selectedOptionConfirmation);
		int characterIndex = _characterIndex;
		bool flag = ((characterIndex == 0 || characterIndex == 3) ? true : false);
		bool flag2 = flag;
		_selectedOptionConfirmation.GlobalPosition = GetRestSiteOptionAnchor().GlobalPosition;
		_selectedOptionConfirmation.Position += _multiplayerConfirmationOffset + (flag2 ? _multiplayerConfirmationFlipOffset : Vector2.Zero);
	}

	public void Shake()
	{
		TaskHelper.RunSafely(DoShake());
	}

	private async Task DoShake()
	{
		ScreenPunchInstance shake = new ScreenPunchInstance(15f, 0.4, 0f);
		Vector2 originalPosition = base.Position;
		while (!shake.IsDone)
		{
			await ToSignal(GetTree(), SceneTree.SignalName.ProcessFrame);
			Vector2 vector = shake.Update(GetProcessDeltaTime());
			base.Position = originalPosition + vector;
		}
		base.Position = originalPosition;
	}

	private Control GetRestSiteOptionAnchor()
	{
		if (_characterIndex < 2)
		{
			return _leftThoughtAnchor;
		}
		return _rightThoughtAnchor;
	}

	private async Task RemoveThoughtBubbleAfterDelay()
	{
		_thoughtBubbleGoAwayCancellation = new CancellationTokenSource();
		await Cmd.Wait(0.5f, _thoughtBubbleGoAwayCancellation.Token);
		if (!_thoughtBubbleGoAwayCancellation.IsCancellationRequested)
		{
			_thoughtBubbleVfx?.GoAway();
			_thoughtBubbleVfx = null;
		}
	}

	private IEnumerable<Node2D> GetChildSpineNodes()
	{
		foreach (Node2D item in GetChildren().OfType<Node2D>())
		{
			if (!(item.GetClass() != "SpineSprite"))
			{
				yield return item;
			}
		}
	}
}
