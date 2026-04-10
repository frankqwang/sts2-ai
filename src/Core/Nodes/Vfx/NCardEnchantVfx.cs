using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Commands;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.Cards;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NCardEnchantVfx : Node2D
{
	private static readonly StringName _progress = new StringName("progress");

	private Tween? _tween;

	private CancellationTokenSource? _cts;

	private CardModel _cardModel;

	private NCard _cardNode;

	private GpuParticles2D _enchantmentSparkles;

	private TextureRect _enchantmentIcon;

	private MegaLabel _enchantmentLabel;

	private static string ScenePath => SceneHelper.GetScenePath("vfx/vfx_card_enchant");

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>(ScenePath);

	[Export(PropertyHint.None, "")]
	public Curve? EmbossCurve { get; set; }

	public static NCardEnchantVfx? Create(CardModel card)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		if (!LocalContext.IsMine(card))
		{
			return null;
		}
		NCardEnchantVfx nCardEnchantVfx = PreloadManager.Cache.GetScene(ScenePath).Instantiate<NCardEnchantVfx>(PackedScene.GenEditState.Disabled);
		nCardEnchantVfx._cardModel = card;
		return nCardEnchantVfx;
	}

	public override void _Ready()
	{
		_enchantmentSparkles = GetNode<GpuParticles2D>("%EnchantmentAppearSparkles");
		_enchantmentIcon = GetNode<TextureRect>("%EnchantmentInViewport/Icon");
		_enchantmentLabel = GetNode<MegaLabel>("%EnchantmentInViewport/Label");
		_enchantmentIcon.Texture = _cardModel.Enchantment.Icon;
		_enchantmentLabel.SetTextAutoSize(_cardModel.Enchantment.DisplayAmount.ToString());
		_enchantmentLabel.Visible = _cardModel.Enchantment.ShowAmount;
		_cardNode = NCard.Create(_cardModel);
		this.AddChildSafely(_cardNode);
		MoveChild(_cardNode, 0);
		_cardNode.UpdateVisuals(PileType.None, CardPreviewMode.Normal);
		_cardNode.EnchantmentTab.Visible = false;
		_cardNode.EnchantmentVfxOverride.Visible = true;
		Viewport node = GetNode<Viewport>("%EnchantmentViewport");
		_cardNode.EnchantmentVfxOverride.Texture = node.GetTexture();
		TaskHelper.RunSafely(PlayAnimation());
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
		_cts?.Cancel();
		_cts?.Dispose();
		if (_cardNode.IsValid() && IsAncestorOf(_cardNode))
		{
			_cardNode.QueueFreeSafely();
		}
	}

	private async Task PlayAnimation()
	{
		_cts = new CancellationTokenSource();
		((ShaderMaterial)_cardNode.EnchantmentVfxOverride.Material).SetShaderParameter(_progress, 0f);
		_tween = CreateTween();
		SfxCmd.Play("event:/sfx/ui/enchant_shimmer");
		_tween.TweenProperty(_cardNode.EnchantmentVfxOverride, "material:shader_parameter/progress", 1f, 1.0).SetEase(Tween.EaseType.InOut).SetTrans(Tween.TransitionType.Quad);
		_tween.Parallel().TweenCallback(Callable.From(() => _enchantmentSparkles.Emitting = true)).SetDelay(0.20000000298023224);
		_tween.Parallel().TweenProperty(_enchantmentSparkles, "position:x", _enchantmentSparkles.Position.X + 72f, 0.4000000059604645).SetDelay(0.20000000298023224);
		await ToSignal(_tween, Tween.SignalName.Finished);
		await Cmd.Wait(1f, _cts.Token);
		CardModel model = _cardNode.Model;
		if (_cardNode.IsInsideTree() && model.Pile == null)
		{
			_tween = CreateTween();
			_tween.TweenProperty(this, "scale", Vector2.Zero, 0.15000000596046448);
			await ToSignal(_tween, Tween.SignalName.Finished);
		}
		else if (_cardNode.IsInsideTree())
		{
			Vector2 targetPosition = model.Pile.Type.GetTargetPosition(_cardNode);
			NCardFlyVfx nCardFlyVfx = NCardFlyVfx.Create(_cardNode, targetPosition, isAddingToPile: false, model.Owner.Character.TrailPath);
			NRun.Instance?.GlobalUi.TopBar.TrailContainer.AddChildSafely(nCardFlyVfx);
			if (nCardFlyVfx.SwooshAwayCompletion != null)
			{
				await nCardFlyVfx.SwooshAwayCompletion.Task;
			}
		}
		this.QueueFreeSafely();
	}
}
