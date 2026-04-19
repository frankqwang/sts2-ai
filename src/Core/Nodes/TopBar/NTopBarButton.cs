using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;

namespace MegaCrit.Sts2.Core.Nodes.TopBar;

public abstract partial class NTopBarButton : NButton
{
	private static readonly StringName _v = new StringName("v");

	protected Control _icon;

	protected ShaderMaterial? _hsv;

	private const float _hoverAngle = -(float)Math.PI / 15f;

	private const float _hoverShaderV = 1.1f;

	protected const float _hoverAnimDur = 0.5f;

	protected static readonly Vector2 _hoverScale = Vector2.One * 1.1f;

	private CancellationTokenSource? _hoverAnimCancelToken;

	private const float _defaultV = 1f;

	protected const float _unhoverAnimDur = 1f;

	private CancellationTokenSource? _unhoverAnimCancelToken;

	private const float _pressDownV = 0.4f;

	protected const float _pressDownDur = 0.25f;

	private CancellationTokenSource? _pressDownCancelToken;

	protected bool IsScreenOpen { get; private set; }

	public override void _Ready()
	{
		ConnectSignals();
		_icon = GetNode<Control>("Control/Icon");
		_hsv = (ShaderMaterial)_icon.Material;
	}

	protected void InitTopBarButton()
	{
		ConnectSignals();
		_icon = GetNode<Control>("Control/Icon");
		_hsv = (ShaderMaterial)_icon.Material;
	}

	public override void _ExitTree()
	{
		base._ExitTree();
		CancelAnimations();
	}

	protected override void OnRelease()
	{
		_pressDownCancelToken?.Cancel();
		_hoverAnimCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimHover(_hoverAnimCancelToken));
	}

	protected override void OnPress()
	{
		base.OnPress();
		_hoverAnimCancelToken?.Cancel();
		_pressDownCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimPressDown(_pressDownCancelToken));
	}

	protected virtual async Task AnimPressDown(CancellationTokenSource cancelToken)
	{
		if (_icon.IsValid())
		{
			float num = 0f;
			float startAngle = _icon.Rotation;
			float targetAngle = startAngle + (float)Math.PI * 2f / 15f;
			while (num < 0.25f)
			{
				_icon.Rotation = Mathf.LerpAngle(startAngle, targetAngle, Ease.CubicOut(num / 0.25f));
				_hsv?.SetShaderParameter(_v, Mathf.Lerp(1.1f, 0.4f, Ease.CubicOut(num / 0.25f)));
				float num2 = num;
				num = num2 + await this.AwaitProcessFrame(cancelToken.Token);
			}
			_icon.Rotation = targetAngle;
			_hsv?.SetShaderParameter(_v, 0.4f);
		}
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		if (IsScreenOpen)
		{
			_hsv?.SetShaderParameter(_v, 1.1f);
			_icon.Scale = _hoverScale;
			return;
		}
		_hsv?.SetShaderParameter(_v, 1.1f);
		_icon.Scale = _hoverScale;
		_unhoverAnimCancelToken?.Cancel();
		_hoverAnimCancelToken?.Cancel();
		_hoverAnimCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimHover(_hoverAnimCancelToken));
	}

	protected override void OnEnable()
	{
		base.OnEnable();
		base.Modulate = Colors.White;
	}

	protected override void OnDisable()
	{
		base.OnDisable();
		base.Modulate = StsColors.disabledTopBarButton;
	}

	protected virtual async Task AnimHover(CancellationTokenSource cancelToken)
	{
		if (_icon.IsValid())
		{
			float num = 0f;
			float startAngle = _icon.Rotation;
			while (num < 0.5f)
			{
				_icon.Rotation = Mathf.LerpAngle(startAngle, -(float)Math.PI / 15f, Ease.BackOut(num / 0.5f));
				float num2 = num;
				num = num2 + await this.AwaitProcessFrame(cancelToken.Token);
			}
			_icon.Rotation = -(float)Math.PI / 15f;
		}
	}

	protected override void OnUnfocus()
	{
		if (IsScreenOpen)
		{
			_pressDownCancelToken?.Cancel();
			_hsv?.SetShaderParameter(_v, 1f);
			_icon.Scale = Vector2.One;
		}
		else
		{
			_hoverAnimCancelToken?.Cancel();
			_pressDownCancelToken?.Cancel();
			_unhoverAnimCancelToken?.Cancel();
			_unhoverAnimCancelToken = new CancellationTokenSource();
			TaskHelper.RunSafely(AnimUnhover(_unhoverAnimCancelToken));
		}
	}

	protected virtual async Task AnimUnhover(CancellationTokenSource cancelToken)
	{
		if (_icon.IsValid())
		{
			float num = 0f;
			float startAngle = _icon.Rotation;
			while (num < 1f)
			{
				_icon.Rotation = Mathf.LerpAngle(startAngle, 0f, Ease.ElasticOut(num / 1f));
				_hsv?.SetShaderParameter(_v, Mathf.Lerp(1.1f, 1f, Ease.ExpoOut(num / 1f)));
				_icon.Scale = _hoverScale.Lerp(Vector2.One, Ease.ExpoOut(num / 1f));
				float num2 = num;
				num = num2 + await this.AwaitProcessFrame(cancelToken.Token);
			}
			_hsv?.SetShaderParameter(_v, 1f);
			_icon.Rotation = 0f;
			_icon.Scale = Vector2.One;
		}
	}

	protected void UpdateScreenOpen()
	{
		bool flag = IsOpen();
		if (IsScreenOpen != flag)
		{
			IsScreenOpen = flag;
			if (!IsScreenOpen)
			{
				OnScreenClosed();
			}
		}
	}

	private void OnScreenClosed()
	{
		CancelAnimations();
		_unhoverAnimCancelToken = new CancellationTokenSource();
		TaskHelper.RunSafely(AnimUnhover(_unhoverAnimCancelToken));
	}

	private void CancelAnimations()
	{
		_hoverAnimCancelToken?.Cancel();
		_pressDownCancelToken?.Cancel();
		_unhoverAnimCancelToken?.Cancel();
	}

	protected abstract bool IsOpen();
}
