using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;

namespace MegaCrit.Sts2.Core.Nodes.Multiplayer;

public partial class NRemoteMouseCursor : Control
{
	private const string _scenePath = "ui/multiplayer/remote_mouse_cursor";

	private TextureRect _textureRect;

	private Vector2? _previousPosition;

	private Vector2? _nextPosition;

	private ulong _lastPositionUpdateMsec;

	private Vector2 _defaultHotspot;

	private Vector2 _drawingHotspot;

	private Vector2 _erasingHotspot;

	[Export(PropertyHint.None, "")]
	private Image _defaultCursorImage;

	[Export(PropertyHint.None, "")]
	private Image _tiltedCursorImage;

	[Export(PropertyHint.None, "")]
	private Image _defaultDrawingImage;

	[Export(PropertyHint.None, "")]
	private Image _tiltedDrawingImage;

	[Export(PropertyHint.None, "")]
	private Image _defaultErasingImage;

	[Export(PropertyHint.None, "")]
	private Image _tiltedErasingImage;

	private ImageTexture _defaultCursorTexture;

	private ImageTexture _tiltedCursorTexture;

	private ImageTexture _defaultDrawingTexture;

	private ImageTexture _tiltedDrawingTexture;

	private ImageTexture _defaultErasingTexture;

	private ImageTexture _tiltedErasingTexture;

	private DrawingMode _drawingMode;

	public ulong PlayerId { get; private set; }

	public static NRemoteMouseCursor Create(ulong playerId)
	{
		NRemoteMouseCursor nRemoteMouseCursor = PreloadManager.Cache.GetAsset<PackedScene>(SceneHelper.GetScenePath("ui/multiplayer/remote_mouse_cursor")).Instantiate<NRemoteMouseCursor>(PackedScene.GenEditState.Disabled);
		nRemoteMouseCursor.PlayerId = playerId;
		return nRemoteMouseCursor;
	}

	public override void _Ready()
	{
		_textureRect = GetNode<TextureRect>("TextureRect");
		_defaultHotspot = -_textureRect.Position;
		_drawingHotspot = NMapDrawings.drawingCursorHotspot;
		_erasingHotspot = NMapDrawings.erasingCursorHotspot;
		base.ProcessMode = ProcessModeEnum.Disabled;
		_defaultCursorTexture = ImageTexture.CreateFromImage(_defaultCursorImage);
		_tiltedCursorTexture = ImageTexture.CreateFromImage(_tiltedCursorImage);
		_defaultDrawingTexture = ImageTexture.CreateFromImage(_defaultDrawingImage);
		_tiltedDrawingTexture = ImageTexture.CreateFromImage(_tiltedDrawingImage);
		_defaultErasingTexture = ImageTexture.CreateFromImage(_defaultErasingImage);
		_tiltedErasingTexture = ImageTexture.CreateFromImage(_tiltedErasingImage);
		GetViewport().Connect(Viewport.SignalName.SizeChanged, Callable.From(RefreshSize));
	}

	public void SetNextPosition(Vector2 position)
	{
		if (!_nextPosition.HasValue)
		{
			_nextPosition = position;
		}
		_previousPosition = _nextPosition;
		_nextPosition = position;
		_lastPositionUpdateMsec = Time.GetTicksMsec();
		base.ProcessMode = ProcessModeEnum.Inherit;
	}

	public override void _Process(double delta)
	{
		if (_previousPosition.HasValue && _nextPosition.HasValue)
		{
			float num = (float)(Time.GetTicksMsec() - _lastPositionUpdateMsec) / 50f;
			base.Position = _previousPosition.Value.Lerp(_nextPosition.Value, Mathf.Clamp(num, 0f, 1f));
			if (num >= 1f)
			{
				base.ProcessMode = ProcessModeEnum.Disabled;
			}
		}
	}

	public void UpdateImage(bool isDown, DrawingMode drawingMode)
	{
		_textureRect.Texture = GetTexture(isDown, drawingMode);
		_drawingMode = drawingMode;
		RefreshSize();
	}

	private Vector2 GetHotspot(DrawingMode drawingMode)
	{
		switch (drawingMode)
		{
		case DrawingMode.None:
			return -_defaultHotspot;
		case DrawingMode.Drawing:
			return -_drawingHotspot;
		case DrawingMode.Erasing:
			return -_erasingHotspot;
		default:
		{
			throw new System.Runtime.CompilerServices.SwitchExpressionException(drawingMode);
			Vector2 result = default(Vector2);
			return result;
		}
		}
	}

	private Texture2D GetTexture(bool isDown, DrawingMode drawingMode)
	{
		switch (drawingMode)
		{
		case DrawingMode.None:
			return isDown ? _tiltedCursorTexture : _defaultCursorTexture;
		case DrawingMode.Drawing:
			return isDown ? _defaultDrawingTexture : _tiltedDrawingTexture;
		case DrawingMode.Erasing:
			return isDown ? _defaultErasingTexture : _tiltedErasingTexture;
		default:
		{
			throw new System.Runtime.CompilerServices.SwitchExpressionException(drawingMode);
			ImageTexture result = default(ImageTexture);
			return result;
		}
		}
	}

	public void RefreshSize()
	{
		if (OS.GetName() == "Windows")
		{
			int num = DisplayServer.ScreenGetDpi();
			float num2 = (float)num / 96f;
			Vector2 vector = GetViewport().GetStretchTransform().Scale.Inverse();
			Vector2 size = _textureRect.Texture.GetSize();
			Vector2 size2 = size * vector * num2;
			_textureRect.Size = size2;
			_textureRect.Position = GetHotspot(_drawingMode) * vector * num2;
		}
		else
		{
			_textureRect.Size = _textureRect.Texture.GetSize();
			_textureRect.Position = GetHotspot(_drawingMode);
		}
	}
}
