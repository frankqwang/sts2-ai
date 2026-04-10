using System;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models.Powers;
using MegaCrit.Sts2.Core.Nodes.Vfx;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Combat;

public partial class NHealthBar : Control
{
	private Control _hpForegroundContainer;

	private Control _hpForeground;

	private Control _poisonForeground;

	private Control _doomForeground;

	private Control _hpMiddleground;

	private MegaLabel _hpLabel;

	private Control _blockContainer;

	private MegaLabel _blockLabel;

	private Control _blockOutline;

	private Creature _creature;

	private Creature? _blockTrackingCreature;

	private readonly LocString _healthBarDead = new LocString("gameplay_ui", "HEALTH_BAR.DEAD");

	private TextureRect _infinityTex;

	private Tween? _blockTween;

	private Tween? _hpLabelFadeTween;

	private Tween? _middlegroundTween;

	private Vector2 _originalBlockPosition;

	private int _currentHpOnLastRefresh = -1;

	private int _maxHpOnLastRefresh = -1;

	private float _expectedMaxFgWidth = -1f;

	private const float _minSize = 12f;

	private static readonly Vector2 _blockAnimOffset = new Vector2(0f, 20f);

	private static readonly Color _defaultFontColor = StsColors.cream;

	private static readonly Color _defaultFontOutlineColor = new Color("900000");

	private static readonly Color _blockOutlineColor = new Color("1B3045");

	private static readonly Color _redForegroundColor = new Color("F1373E");

	private static readonly Color _blockHpForegroundColor = new Color("3B6FA3");

	private static readonly Color _invincibleForegroundColor = new Color("C5BBED");

	private const float _foregroundContainerInset = 10f;

	private float MaxFgWidth
	{
		get
		{
			if (!(_expectedMaxFgWidth > 0f))
			{
				return _hpForegroundContainer.Size.X;
			}
			return _expectedMaxFgWidth;
		}
	}

	public Control HpBarContainer { get; private set; }

	public void SetCreature(Creature creature)
	{
		if (_creature != null)
		{
			throw new InvalidOperationException("Creature was already set.");
		}
		_creature = creature;
		_hpForeground.OffsetRight = GetFgWidth(_creature.CurrentHp) - MaxFgWidth;
		_hpMiddleground.OffsetRight = _hpForeground.OffsetRight - 2f;
	}

	public override void _Ready()
	{
		HpBarContainer = GetNode<Control>("%HpBarContainer");
		_hpForegroundContainer = GetNode<Control>("%HpForegroundContainer");
		_hpMiddleground = GetNode<Control>("%HpMiddleground");
		_hpForeground = GetNode<Control>("%HpForeground");
		_poisonForeground = GetNode<Control>("%PoisonForeground");
		_doomForeground = GetNode<Control>("%DoomForeground");
		_hpLabel = GetNode<MegaLabel>("%HpLabel");
		_blockContainer = GetNode<Control>("%BlockContainer");
		_blockLabel = GetNode<MegaLabel>("%BlockLabel");
		_blockOutline = GetNode<Control>("%BlockOutline");
		_infinityTex = GetNode<TextureRect>("%InfinityTex");
		_originalBlockPosition = _blockContainer.Position;
	}

	private void DebugToggleVisibility()
	{
		base.Visible = !NCombatUi.IsDebugHidingHpBar;
	}

	public void UpdateLayoutForCreatureBounds(Control bounds)
	{
		float valueOrDefault = (24f - _creature.Monster?.HpBarSizeReduction).GetValueOrDefault();
		HpBarContainer.GlobalPosition = new Vector2(bounds.GlobalPosition.X - valueOrDefault * 0.5f, HpBarContainer.GlobalPosition.Y);
		float x = bounds.Size.X + valueOrDefault;
		SetHpBarContainerSizeWithOffsets(new Vector2(x, HpBarContainer.Size.Y));
		float num = _blockContainer.Size.X * 0.5f;
		_blockContainer.GlobalPosition = new Vector2(bounds.GlobalPosition.X - num, _blockContainer.GlobalPosition.Y);
		_originalBlockPosition = _blockContainer.Position;
	}

	public void UpdateWidthRelativeToReferenceValue(float refMaxHp, float refWidth)
	{
		Vector2 size = HpBarContainer.Size;
		size.X = (float)_creature.MaxHp / refMaxHp * refWidth;
		SetHpBarContainerSizeWithOffsetsImmediately(size);
	}

	private void SetHpBarContainerSizeWithOffsets(Vector2 size)
	{
		Callable.From(delegate
		{
			SetHpBarContainerSizeWithOffsetsImmediately(size);
		}).CallDeferred();
	}

	private void SetHpBarContainerSizeWithOffsetsImmediately(Vector2 size)
	{
		if (!HpBarContainer.Size.IsEqualApprox(size))
		{
			_middlegroundTween?.Kill();
			HpBarContainer.Size = size;
			_expectedMaxFgWidth = size.X - 10f;
			_hpForeground.OffsetRight = GetFgWidth(_creature.CurrentHp, _expectedMaxFgWidth) - _expectedMaxFgWidth;
			_hpMiddleground.OffsetRight = _hpForeground.OffsetRight - 2f;
		}
	}

	public void RefreshValues()
	{
		RefreshBlockUi();
		RefreshForeground();
		RefreshMiddleground();
		RefreshText();
	}

	private void RefreshMiddleground()
	{
		if (_creature.CurrentHp <= 0)
		{
			_hpMiddleground.Visible = false;
			return;
		}
		_hpMiddleground.Visible = true;
		Control hpMiddleground = _hpMiddleground;
		Vector2 position = _hpMiddleground.Position;
		position.X = 1f;
		hpMiddleground.Position = position;
		int currentHp = _creature.CurrentHp;
		int maxHp = _creature.MaxHp;
		if (currentHp != _currentHpOnLastRefresh || maxHp != _maxHpOnLastRefresh)
		{
			_currentHpOnLastRefresh = currentHp;
			_maxHpOnLastRefresh = maxHp;
			float num = (_creature.HasPower<PoisonPower>() ? _poisonForeground.OffsetRight : _hpForeground.OffsetRight);
			bool flag = num >= _hpMiddleground.OffsetRight;
			_hpMiddleground.OffsetRight += 1f;
			_middlegroundTween?.Kill();
			_middlegroundTween = CreateTween();
			_middlegroundTween.TweenProperty(_hpMiddleground, "offset_right", num - 2f, 1.0).SetDelay(flag ? 0.0 : 1.0).SetEase(Tween.EaseType.Out)
				.SetTrans(Tween.TransitionType.Expo);
		}
	}

	private void RefreshForeground()
	{
		if (_creature.CurrentHp <= 0)
		{
			_poisonForeground.Visible = false;
			_doomForeground.Visible = false;
			_hpForeground.Visible = false;
			return;
		}
		_hpForeground.Visible = true;
		float offsetRight = GetFgWidth(_creature.CurrentHp) - MaxFgWidth;
		_hpForeground.OffsetRight = offsetRight;
		if (_creature.ShowsInfiniteHp)
		{
			_hpForeground.SelfModulate = _invincibleForegroundColor;
			return;
		}
		int powerAmount = _creature.GetPowerAmount<DoomPower>();
		int num = _creature.GetPower<PoisonPower>()?.CalculateTotalDamageNextTurn() ?? 0;
		if (_creature.HasPower<PoisonPower>())
		{
			if (num > 0)
			{
				_poisonForeground.Visible = true;
				if (IsPoisonLethal(num))
				{
					_poisonForeground.OffsetLeft = 0f;
					_poisonForeground.OffsetRight = offsetRight;
					_hpForeground.Visible = false;
				}
				else
				{
					float fgWidth = GetFgWidth(_creature.CurrentHp - num);
					_hpForeground.OffsetRight = fgWidth - MaxFgWidth;
					_hpForeground.Visible = true;
					int patchMarginLeft = ((NinePatchRect)_poisonForeground).PatchMarginLeft;
					_poisonForeground.OffsetLeft = Math.Max(0f, fgWidth - (float)patchMarginLeft);
					_poisonForeground.OffsetRight = offsetRight;
				}
			}
			else
			{
				_poisonForeground.Visible = false;
			}
		}
		else
		{
			_poisonForeground.Visible = false;
			_poisonForeground.OffsetLeft = 0f;
		}
		if (_creature.HasPower<DoomPower>())
		{
			if (powerAmount > 0)
			{
				_doomForeground.Visible = true;
				float num2 = GetFgWidth(powerAmount) - MaxFgWidth;
				if (IsDoomLethal(powerAmount, num))
				{
					if (!IsPoisonLethal(num))
					{
						_doomForeground.OffsetRight = _hpForeground.OffsetRight;
						_hpForeground.Visible = false;
					}
					else
					{
						_hpForeground.Visible = false;
						_doomForeground.Visible = false;
					}
				}
				else
				{
					int patchMarginRight = ((NinePatchRect)_doomForeground).PatchMarginRight;
					_doomForeground.OffsetRight = Math.Min(0f, num2 + (float)patchMarginRight);
					_hpForeground.Visible = true;
				}
			}
			else
			{
				_doomForeground.Visible = false;
			}
		}
		else
		{
			_doomForeground.Visible = false;
		}
	}

	private void RefreshBlockUi()
	{
		if (_creature.Block <= 0)
		{
			Creature blockTrackingCreature = _blockTrackingCreature;
			if (blockTrackingCreature == null || blockTrackingCreature.Block <= 0)
			{
				if (_blockContainer.Visible)
				{
					NBlockBrokenVfx nBlockBrokenVfx = NBlockBrokenVfx.Create();
					if (nBlockBrokenVfx != null)
					{
						this.AddChildSafely(nBlockBrokenVfx);
						nBlockBrokenVfx.GlobalPosition = _blockContainer.GlobalPosition + _blockContainer.Size * 0.5f;
					}
				}
				_blockContainer.Visible = false;
				_blockOutline.Visible = false;
				_hpForeground.SelfModulate = _redForegroundColor;
				return;
			}
		}
		_blockOutline.Visible = true;
		_hpForeground.SelfModulate = _blockHpForegroundColor;
		if (_creature.Block > 0)
		{
			_blockContainer.Visible = true;
			_blockLabel.SetTextAutoSize(_creature.Block.ToString());
		}
	}

	private void RefreshText()
	{
		if (_creature.CurrentHp <= 0)
		{
			_hpLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, _defaultFontColor);
			_hpLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, _defaultFontOutlineColor);
			_hpLabel.SetTextAutoSize(_healthBarDead.GetRawText());
			return;
		}
		if (_creature.ShowsInfiniteHp)
		{
			_infinityTex.Visible = _creature.IsAlive;
			_doomForeground.Modulate = Colors.Transparent;
			_hpLabel.Visible = !_infinityTex.Visible;
			return;
		}
		_hpLabel.Visible = true;
		int poisonDamage = _creature.GetPower<PoisonPower>()?.CalculateTotalDamageNextTurn() ?? 0;
		int powerAmount = _creature.GetPowerAmount<DoomPower>();
		Color color;
		Color color2;
		if (IsPoisonLethal(poisonDamage))
		{
			color = new Color("76FF40");
			color2 = new Color("074700");
		}
		else if (IsDoomLethal(powerAmount, poisonDamage))
		{
			color = new Color("FB8DFF");
			color2 = new Color("2D1263");
		}
		else
		{
			if (_creature.Block <= 0)
			{
				Creature blockTrackingCreature = _blockTrackingCreature;
				if (blockTrackingCreature == null || blockTrackingCreature.Block <= 0)
				{
					color = _defaultFontColor;
					color2 = _defaultFontOutlineColor;
					goto IL_0151;
				}
			}
			color = _defaultFontColor;
			color2 = _blockOutlineColor;
		}
		goto IL_0151;
		IL_0151:
		_hpLabel.AddThemeColorOverride(ThemeConstants.Label.fontColor, color);
		_hpLabel.AddThemeColorOverride(ThemeConstants.Label.fontOutlineColor, color2);
		_hpLabel.SetTextAutoSize($"{_creature.CurrentHp}/{_creature.MaxHp}");
	}

	private bool IsPoisonLethal(int poisonDamage)
	{
		if (poisonDamage <= 0 || !_creature.HasPower<PoisonPower>())
		{
			return false;
		}
		return poisonDamage >= _creature.CurrentHp;
	}

	private bool IsDoomLethal(int doomAmount, int poisonDamage)
	{
		if (doomAmount <= 0 || !_creature.HasPower<DoomPower>())
		{
			return false;
		}
		return doomAmount >= _creature.CurrentHp - poisonDamage;
	}

	private float GetFgWidth(int amount)
	{
		return GetFgWidth(amount, MaxFgWidth);
	}

	private float GetFgWidth(int amount, float maxFgWidth)
	{
		if (_creature.MaxHp <= 0)
		{
			return 0f;
		}
		float val = (float)amount / (float)_creature.MaxHp * maxFgWidth;
		return Math.Max(val, (_creature.CurrentHp > 0) ? 12f : 0f);
	}

	public void FadeOutHpLabel(float duration, float finalAlpha)
	{
		_hpLabelFadeTween?.Kill();
		_hpLabelFadeTween = CreateTween();
		_hpLabelFadeTween.TweenProperty(_hpLabel, "modulate:a", finalAlpha, duration).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Expo);
	}

	public void FadeInHpLabel(float duration)
	{
		_hpLabelFadeTween?.Kill();
		_hpLabelFadeTween = CreateTween();
		_hpLabelFadeTween.TweenProperty(_hpLabel, "modulate:a", 1f, duration);
	}

	public void AnimateInBlock(int oldBlock, int blockGain)
	{
		if (oldBlock == 0 && blockGain != 0)
		{
			_blockContainer.Visible = true;
			if (SaveManager.Instance.PrefsSave.FastMode != FastModeType.Instant)
			{
				_blockContainer.Modulate = StsColors.transparentWhite;
				_blockContainer.Position = _originalBlockPosition - _blockAnimOffset;
				_blockTween?.Kill();
				_blockTween = CreateTween().SetParallel();
				_blockTween.TweenProperty(_blockContainer, "modulate:a", 1f, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Sine);
				_blockTween.TweenProperty(_blockContainer, "position", _originalBlockPosition, 0.5).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Back);
			}
		}
	}

	public void TrackBlockStatus(Creature creature)
	{
		_blockTrackingCreature = creature;
	}
}
