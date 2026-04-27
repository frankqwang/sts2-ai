using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NAiDecisionOverlay : CanvasLayer
{
	private const float PanelWidth = 488f;
	private const float PanelHeight = 232f;
	private const float MarginLeft = 28f;
	private const float MarginTop = 190f;
	private const float MinX = 12f;
	private const float MinY = 12f;
	private const float ContentMarginX = 18f;
	private const float ContentMarginY = 14f;

	private sealed class OverlayPayload
	{
		public string? title { get; set; }
		public string? state_type { get; set; }
		public int? step { get; set; }
		public string? action_source { get; set; }
		public JsonElement? chosen_action { get; set; }
		public string? chosen_action_text { get; set; }
		public string? reason { get; set; }
	}

	public string OverlayFilePath { get; set; } = "";

	private Panel _panel = null!;
	private RichTextLabel _label = null!;
	private double _nextPollAt;
	private long _lastModifiedTicks = -1;
	private Vector2 _lastViewportSize = Vector2.Zero;

	private bool _initialized;

	/// <summary>
	/// Manual init — Godot does NOT call _Ready/_Process on C# nodes
	/// loaded from external mod assemblies. Called explicitly after AddChild.
	/// </summary>
	public void Initialize()
	{
		if (_initialized) return;
		_initialized = true;

		try
		{
			Layer = 100;
			ProcessMode = ProcessModeEnum.Always;
			Visible = true;

			_panel = new Panel();
			_panel.MouseFilter = Control.MouseFilterEnum.Ignore;
			_panel.Visible = true;
			_panel.Position = new Vector2(MarginLeft, MarginTop);
			_panel.Size = new Vector2(PanelWidth, PanelHeight);
			_panel.CustomMinimumSize = Vector2.Zero;
			_panel.SetAnchorsPreset(Control.LayoutPreset.TopLeft);
			AddChild(_panel);

			var panelStyle = new StyleBoxFlat();
			panelStyle.BgColor = new Color(0.03f, 0.05f, 0.08f, 0.62f);
			panelStyle.BorderColor = new Color(0.96f, 0.83f, 0.45f, 0.21f);
			panelStyle.SetBorderWidthAll(1);
			panelStyle.SetCornerRadiusAll(10);
			panelStyle.ShadowColor = new Color(0f, 0f, 0f, 0.25f);
			panelStyle.ShadowSize = 10;
			_panel.AddThemeStyleboxOverride("panel", panelStyle);

			_label = new RichTextLabel();
			_label.BbcodeEnabled = true;
			_label.ScrollActive = false;
			_label.FitContent = false;
			_label.ClipContents = true;
			_label.AutowrapMode = TextServer.AutowrapMode.WordSmart;
			_label.MouseFilter = Control.MouseFilterEnum.Ignore;
			_label.Position = new Vector2(ContentMarginX, ContentMarginY);
			_label.Size = new Vector2(PanelWidth - (ContentMarginX * 2f), PanelHeight - (ContentMarginY * 2f));
			_label.CustomMinimumSize = Vector2.Zero;
			_label.AddThemeFontSizeOverride("normal_font_size", 14);
			_label.AddThemeConstantOverride("line_separation", 3);
			_label.AddThemeColorOverride("default_color", new Color(0.96f, 0.98f, 1.00f, 1.0f));
			_panel.AddChild(_label);

			SetPlaceholderText();
			UpdatePanelLayout(force: true);

			GD.Print($"[NAiDecisionOverlay] initialized, panel at {_panel.Position}");
		}
		catch (Exception ex)
		{
			GD.PrintErr($"[NAiDecisionOverlay] Initialize FAILED: {ex}");
		}
	}

	/// <summary>
	/// Called manually from McpMod.ProcessMainThreadQueue each frame,
	/// since Godot won't call _Process on external mod assemblies.
	/// </summary>
	public void ManualProcess()
	{
		if (!_initialized) return;
		UpdatePanelLayout();

		if (string.IsNullOrWhiteSpace(OverlayFilePath))
		{
			return;
		}

		double now = Time.GetTicksMsec() / 1000.0;
		if (now < _nextPollAt)
		{
			return;
		}

		_nextPollAt = now + 0.03;

		try
		{
			FileInfo fileInfo = new FileInfo(OverlayFilePath);
			if (!fileInfo.Exists)
			{
				SetPlaceholderText();
				return;
			}

			long modifiedTicks = fileInfo.LastWriteTimeUtc.Ticks;
			if (modifiedTicks == _lastModifiedTicks)
			{
				return;
			}

			_lastModifiedTicks = modifiedTicks;
			using System.IO.FileStream stream = new System.IO.FileStream(
				OverlayFilePath,
				System.IO.FileMode.Open,
				System.IO.FileAccess.Read,
				System.IO.FileShare.ReadWrite | System.IO.FileShare.Delete
			);
			using System.IO.StreamReader reader = new System.IO.StreamReader(stream, Encoding.UTF8);
			string raw = reader.ReadToEnd();
			if (string.IsNullOrWhiteSpace(raw))
			{
				SetPlaceholderText();
				return;
			}

			OverlayPayload? payload = JsonSerializer.Deserialize<OverlayPayload>(raw);
			if (payload != null)
			{
				_label.Text = BuildText(payload);
			}
		}
		catch (Exception)
		{
			// Overlay is best-effort only; never affect game logic.
		}
	}

	private void UpdatePanelLayout(bool force = false)
	{
		if (!GodotObject.IsInstanceValid(_panel))
		{
			return;
		}

		Vector2 viewportSize = GetViewport()?.GetVisibleRect().Size ?? Vector2.Zero;
		if (!force && viewportSize == _lastViewportSize)
		{
			return;
		}

		_lastViewportSize = viewportSize;

		float width = PanelWidth;
		float height = PanelHeight;
		if (viewportSize.X > 0)
		{
			width = MathF.Min(PanelWidth, MathF.Max(280f, viewportSize.X - (MinX * 2f)));
		}
		if (viewportSize.Y > 0)
		{
			height = MathF.Min(PanelHeight, MathF.Max(110f, viewportSize.Y - (MinY * 2f)));
		}

		_panel.Size = new Vector2(width, height);
		_panel.CustomMinimumSize = Vector2.Zero;

		float x = viewportSize.X > 0 ? MathF.Min(MarginLeft, MathF.Max(MinX, viewportSize.X - width - MinX)) : MinX;
		float y = viewportSize.Y > 0 ? MathF.Min(MarginTop, MathF.Max(MinY, viewportSize.Y - height - MinY)) : MinY;
		x = MathF.Max(MinX, x);
		y = MathF.Max(MinY, y);
		_panel.Position = new Vector2(x, y);
		if (GodotObject.IsInstanceValid(_label))
		{
			_label.Position = new Vector2(ContentMarginX, ContentMarginY);
			_label.Size = new Vector2(
				MathF.Max(10f, width - (ContentMarginX * 2f)),
				MathF.Max(10f, height - (ContentMarginY * 2f)));
		}
	}

	private void SetPlaceholderText()
	{
		_label.Text =
			"[font_size=16][b][color=#ffe3a1]AI[/color][/b][/font_size] [color=#9fc1df]waiting[/color]\n" +
			"[color=#ffffff]choice: -[/color]\n" +
			"[color=#d8e7f4]reason: waiting for model decision[/color]";
	}

	private static string BuildText(OverlayPayload payload)
	{
		string title = string.IsNullOrWhiteSpace(payload.title) ? "AI Decision" : payload.title!;
		string stateType = PrettyStateName(payload.state_type);
		string chosenAction = ResolveChosenActionText(payload);
		string source = string.IsNullOrWhiteSpace(payload.action_source) ? "-" : payload.action_source!;
		string reason = string.IsNullOrWhiteSpace(payload.reason) ? "" : payload.reason!;
		string step = payload.step?.ToString(CultureInfo.InvariantCulture) ?? "-";

		StringBuilder sb = new StringBuilder();
		sb.Append("[font_size=15][b][color=#ffe3a1]").Append(EscapeBbcode(title)).Append("[/color][/b][/font_size]");
		sb.Append(" [color=#9fc1df]step=").Append(EscapeBbcode(step));
		sb.Append(" source=").Append(EscapeBbcode(source));
		sb.Append(" state=").Append(EscapeBbcode(stateType)).AppendLine("[/color]");

		sb.Append("[font_size=18][b][color=#ffffff]choice: ");
		sb.Append(EscapeBbcode(Truncate(chosenAction, 88))).AppendLine("[/color][/b][/font_size]");

		sb.Append("[color=#d8e7f4]reason: ");
		sb.Append(EscapeBbcode(Truncate(string.IsNullOrWhiteSpace(reason) ? "-" : reason, 180)));
		sb.Append("[/color]");

		return sb.ToString().TrimEnd();
	}

	private static string Truncate(string text, int maxChars)
	{
		if (string.IsNullOrEmpty(text) || text.Length <= maxChars)
		{
			return text;
		}
		return text.Substring(0, Math.Max(0, maxChars - 1)).TrimEnd() + "…";
	}

	private static string EscapeBbcode(string text)
	{
		return text.Replace("[", "［").Replace("]", "］");
	}

	private static string PrettyStateName(string? raw)
	{
		string state = string.IsNullOrWhiteSpace(raw) ? "unknown" : raw.Trim().ToLowerInvariant();
		return state switch
		{
			"map" => "Route",
			"event" => "Event",
			"shop" => "Shop",
			"treasure" => "Treasure",
			"rest_site" => "Rest Site",
			"combat" => "Combat",
			"monster" => "Combat",
			"elite" => "Elite",
			"boss" => "Boss",
			"combat_rewards" => "Combat Rewards",
			"card_reward" => "Card Reward",
			"card_select" => "Card Select",
			"hand_select" => "Hand Select",
			"game_over" => "Game Over",
			_ => state,
		};
	}

	private static string ResolveChosenActionText(OverlayPayload payload)
	{
		if (!string.IsNullOrWhiteSpace(payload.chosen_action_text))
		{
			return payload.chosen_action_text!;
		}
		if (!payload.chosen_action.HasValue)
		{
			return "-";
		}
		JsonElement elem = payload.chosen_action.Value;
		if (elem.ValueKind == JsonValueKind.String)
		{
			return elem.GetString() ?? "-";
		}
		if (elem.ValueKind == JsonValueKind.Object)
		{
			if (elem.TryGetProperty("action", out JsonElement actionElem) && actionElem.ValueKind == JsonValueKind.String)
			{
				string action = actionElem.GetString() ?? "action";
				string card = elem.TryGetProperty("card_id", out JsonElement cardElem) && cardElem.ValueKind == JsonValueKind.String
					? cardElem.GetString() ?? ""
					: "";
				string target = elem.TryGetProperty("target_id", out JsonElement targetElem) && targetElem.ValueKind == JsonValueKind.Number
					? " -> enemy" + targetElem.GetInt32().ToString(CultureInfo.InvariantCulture)
					: "";
				return string.IsNullOrWhiteSpace(card) ? action : action + " " + card + target;
			}
		}
		return elem.ToString();
	}

}
