using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NAiDecisionOverlay : CanvasLayer
{
	private const float PanelWidth = 438f;
	private const float PanelHeight = 442f;
	private const float MarginRight = 18f;
	private const float MarginTop = 74f;
	private const float MinX = 12f;
	private const float MinY = 12f;

	private sealed class OverlayPayload
	{
		public string? title { get; set; }
		public string? state_type { get; set; }
		public int? step { get; set; }
		public int? act { get; set; }
		public int? floor { get; set; }
		public string? action_source { get; set; }
		public string? chosen_action { get; set; }
		public string? reason { get; set; }
		public string? reasoning_zh { get; set; }
		public string? reasoning_en { get; set; }
		public string? next_boss { get; set; }
		public string? next_boss_name { get; set; }
		public string? next_boss_archetype { get; set; }
		public double? boss_readiness { get; set; }
		public double? deck_quality { get; set; }
		public double? value { get; set; }
		public Dictionary<string, double>? problem_vector { get; set; }
		public string[]? details { get; set; }
		public OverlayPlayerPayload? player { get; set; }
		public OverlayEnemyPayload[]? enemies { get; set; }
		public OverlayOptionPayload[]? options { get; set; }
		public OverlayRewardPayload? reward_shaping { get; set; }
	}

	private sealed class OverlayPlayerPayload
	{
		public int? hp { get; set; }
		public int? max_hp { get; set; }
		public int? energy { get; set; }
		public int? block { get; set; }
		public int? gold { get; set; }
		public int? deck_size { get; set; }
		public double? deck_score { get; set; }
	}

	private sealed class OverlayEnemyPayload
	{
		public string? name { get; set; }
		public int? hp { get; set; }
		public int? max_hp { get; set; }
		public int? block { get; set; }
		public string? intent { get; set; }
		public int? intent_damage { get; set; }
	}

	private sealed class OverlayOptionPayload
	{
		public string? label { get; set; }
		public double? prob { get; set; }
		public double? advantage { get; set; }
		public int? cost { get; set; }
		public string? target { get; set; }
		public JsonElement? chosen { get; set; }
	}

	private sealed class OverlayRewardPayload
	{
		public double? boss_readiness_score { get; set; }
	}

	public string OverlayFilePath { get; set; } = "";
	public int ControlPort { get; set; } = 8765;

	private PanelContainer _panel = null!;
	private RichTextLabel _label = null!;
	private Button _pauseBtn = null!;
	private Button _stepBtn = null!;
	private Button _resumeBtn = null!;
	private bool _isPaused;
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

			_panel = new PanelContainer();
			_panel.MouseFilter = Control.MouseFilterEnum.Ignore;
			_panel.Visible = true;
			_panel.Position = new Vector2(1920 - PanelWidth - MarginRight, MarginTop);
			_panel.Size = new Vector2(PanelWidth, PanelHeight);
			AddChild(_panel);

			var panelStyle = new StyleBoxFlat();
			panelStyle.BgColor = new Color(0.03f, 0.05f, 0.08f, 0.88f);
			panelStyle.BorderColor = new Color(0.96f, 0.83f, 0.45f, 0.30f);
			panelStyle.SetBorderWidthAll(1);
			panelStyle.SetCornerRadiusAll(16);
			panelStyle.ShadowColor = new Color(0f, 0f, 0f, 0.36f);
			panelStyle.ShadowSize = 10;
			_panel.AddThemeStyleboxOverride("panel", panelStyle);

			var margin = new MarginContainer();
			margin.AddThemeConstantOverride("margin_left", 18);
			margin.AddThemeConstantOverride("margin_right", 18);
			margin.AddThemeConstantOverride("margin_top", 16);
			margin.AddThemeConstantOverride("margin_bottom", 16);
			_panel.AddChild(margin);

			_label = new RichTextLabel();
			_label.BbcodeEnabled = true;
			_label.ScrollActive = false;
			_label.FitContent = true;
			_label.AutowrapMode = TextServer.AutowrapMode.WordSmart;
			_label.MouseFilter = Control.MouseFilterEnum.Ignore;
			_label.AddThemeFontSizeOverride("normal_font_size", 14);
			_label.AddThemeConstantOverride("line_separation", 4);
			_label.AddThemeColorOverride("default_color", new Color(0.96f, 0.98f, 1.00f, 1.0f));
			margin.AddChild(_label);

			SetPlaceholderText();

			// Playback control buttons at bottom of panel
			var btnRow = new HBoxContainer();
			btnRow.MouseFilter = Control.MouseFilterEnum.Stop;
			btnRow.AddThemeConstantOverride("separation", 8);
			AddChild(btnRow);
			btnRow.Position = new Vector2(_panel.Position.X, _panel.Position.Y + PanelHeight + 8);

			_pauseBtn = CreateControlButton("⏸ 暂停", () => SendControlCommand("pause"));
			_stepBtn = CreateControlButton("⏭ 下一步", () => SendControlCommand("step"));
			_resumeBtn = CreateControlButton("▶ 继续", () => SendControlCommand("resume"));
			btnRow.AddChild(_pauseBtn);
			btnRow.AddChild(_stepBtn);
			btnRow.AddChild(_resumeBtn);

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
			height = MathF.Min(PanelHeight, MathF.Max(220f, viewportSize.Y - (MinY * 2f)));
		}

		_panel.Size = new Vector2(width, height);

		float x = viewportSize.X > 0 ? viewportSize.X - width - MarginRight : MinX;
		float y = viewportSize.Y > 0 ? MathF.Min(MarginTop, MathF.Max(MinY, viewportSize.Y - height - MinY)) : MinY;
		x = MathF.Max(MinX, x);
		y = MathF.Max(MinY, y);
		_panel.Position = new Vector2(x, y);
	}

	private void SetPlaceholderText()
	{
		_label.Text =
			"[font_size=18][b][color=#ffe3a1]AI Decision[/color][/b][/font_size]\n" +
			"[color=#9fc1df]Waiting for visible demo data...[/color]";
	}

	private static string BuildText(OverlayPayload payload)
	{
		string title = string.IsNullOrWhiteSpace(payload.title) ? "AI Decision" : payload.title!;
		string stateType = PrettyStateName(payload.state_type);
		string chosenAction = string.IsNullOrWhiteSpace(payload.chosen_action) ? "-" : payload.chosen_action!;

		StringBuilder sb = new StringBuilder();
		// Line 1: Title + screen type
		sb.Append("[font_size=18][b][color=#ffe3a1]").Append(title).Append("[/color][/b][/font_size]");
		sb.Append("  [color=#a9c6de]").Append(stateType).AppendLine("[/color]");

		// Line 2: AI metrics (always shown, fixed height)
		sb.Append("[color=#9fc1df]V=").Append(FormatFloat(payload.value ?? 0, "0.00"));
		sb.Append("  Deck=").Append(FormatFloat(payload.deck_quality ?? 0, "0.00"));
		sb.Append("  Boss=").Append(FormatFloat((payload.boss_readiness ?? 0) * 100, "0")).Append("%");
		sb.AppendLine("[/color]");

		// Line 3: Problem vector (compact bar style)
		if (payload.problem_vector is { Count: > 0 })
		{
			string[] pvKeys = { "frontload", "aoe", "block", "draw", "scaling" };
			string[] pvLabels = { "DMG", "AOE", "BLK", "DRW", "SCL" };
			sb.Append("[color=#7a9ab5]");
			for (int i = 0; i < pvKeys.Length; i++)
			{
				if (payload.problem_vector.TryGetValue(pvKeys[i], out double val))
				{
					if (i > 0) sb.Append("  ");
					sb.Append(pvLabels[i]).Append(':').Append(FormatFloat(val * 100, "0")).Append('%');
				}
			}
			sb.AppendLine("[/color]");
		}

		sb.AppendLine();

		// Line 4: Chosen action (large)
		sb.Append("[font_size=20][b][color=#ffffff]").Append(chosenAction).AppendLine("[/color][/b][/font_size]");

		// Lines 5+: Options
		if (payload.options is { Length: > 0 })
		{
			foreach (OverlayOptionPayload option in payload.options.Take(5))
			{
				bool chosen = ParseChosen(option.chosen);
				sb.Append(chosen ? "[color=#ffd978]> [/color]" : "[color=#55697d]- [/color]");
				sb.Append(string.IsNullOrWhiteSpace(option.label) ? "?" : option.label);
				if (option.prob.HasValue)
				{
					sb.Append("  [color=#9fc1df]").Append(FormatFloat(option.prob.Value * 100, "0")).Append("%[/color]");
				}
				if (!string.IsNullOrWhiteSpace(option.target))
				{
					sb.Append("  [color=#9fc1df]-> ").Append(option.target).Append("[/color]");
				}
				sb.AppendLine();
			}
		}

		return sb.ToString().TrimEnd();
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

	private static string FormatFloat(double value, string format)
	{
		return value.ToString(format, CultureInfo.InvariantCulture);
	}

	private static bool ParseChosen(JsonElement? value)
	{
		if (!value.HasValue)
		{
			return false;
		}

		JsonElement elem = value.Value;
		switch (elem.ValueKind)
		{
			case JsonValueKind.True:
				return true;
			case JsonValueKind.False:
				return false;
			case JsonValueKind.String:
				string text = elem.GetString() ?? "";
				if (bool.TryParse(text, out bool parsed))
				{
					return parsed;
				}
				if (string.Equals(text, "1", StringComparison.Ordinal))
				{
					return true;
				}
				if (string.Equals(text, "0", StringComparison.Ordinal))
				{
					return false;
				}
				return false;
			case JsonValueKind.Number:
				return elem.TryGetInt32(out int intValue) && intValue != 0;
			default:
				return false;
		}
	}

	private static Button CreateControlButton(string text, Action onPress)
	{
		var btn = new Button();
		btn.Text = text;
		btn.MouseFilter = Control.MouseFilterEnum.Stop;
		btn.CustomMinimumSize = new Vector2(120, 32);

		var style = new StyleBoxFlat();
		style.BgColor = new Color(0.12f, 0.16f, 0.22f, 0.92f);
		style.BorderColor = new Color(0.96f, 0.83f, 0.45f, 0.40f);
		style.SetBorderWidthAll(1);
		style.SetCornerRadiusAll(8);
		style.SetContentMarginAll(6);
		btn.AddThemeStyleboxOverride("normal", style);

		var hoverStyle = new StyleBoxFlat();
		hoverStyle.BgColor = new Color(0.18f, 0.24f, 0.32f, 0.95f);
		hoverStyle.BorderColor = new Color(0.96f, 0.83f, 0.45f, 0.60f);
		hoverStyle.SetBorderWidthAll(1);
		hoverStyle.SetCornerRadiusAll(8);
		hoverStyle.SetContentMarginAll(6);
		btn.AddThemeStyleboxOverride("hover", hoverStyle);

		btn.AddThemeColorOverride("font_color", new Color(0.96f, 0.98f, 1.0f));
		btn.AddThemeFontSizeOverride("font_size", 13);
		btn.Pressed += onPress;
		return btn;
	}

	private void SendControlCommand(string command)
	{
		try
		{
			// Write command to a file next to the overlay JSON.
			// Python demo_play.py polls this file each frame.
			if (string.IsNullOrWhiteSpace(OverlayFilePath)) return;
			string cmdFile = Path.Combine(Path.GetDirectoryName(OverlayFilePath)!, "playback.cmd");
			File.WriteAllText(cmdFile, command);
			_isPaused = command == "pause" || command == "step";
			GD.Print($"[NAiDecisionOverlay] {command} written to {cmdFile}");
		}
		catch (Exception ex)
		{
			GD.PrintErr($"[NAiDecisionOverlay] control command failed: {ex.Message}");
		}
	}
}
