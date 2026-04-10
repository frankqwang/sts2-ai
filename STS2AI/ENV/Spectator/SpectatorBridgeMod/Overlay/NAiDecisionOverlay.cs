using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Godot;

namespace MegaCrit.Sts2.Core.Nodes.Debug;

public partial class NAiDecisionOverlay : CanvasLayer
{
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

	private RichTextLabel _label = null!;
	private double _nextPollAt;
	private long _lastModifiedTicks = -1;

	public override void _Ready()
	{
		Layer = 50;
		ProcessMode = ProcessModeEnum.Always;

		Control root = new Control();
		root.SetAnchorsPreset(Control.LayoutPreset.FullRect);
		root.MouseFilter = Control.MouseFilterEnum.Ignore;
		AddChild(root);

		PanelContainer panel = new PanelContainer();
		panel.MouseFilter = Control.MouseFilterEnum.Ignore;
		panel.AnchorLeft = 1.0f;
		panel.AnchorTop = 0.0f;
		panel.AnchorRight = 1.0f;
		panel.AnchorBottom = 0.0f;
		panel.OffsetLeft = -456f;
		panel.OffsetTop = 74f;
		panel.OffsetRight = -18f;
		panel.OffsetBottom = 516f;
		root.AddChild(panel);

		StyleBoxFlat panelStyle = new StyleBoxFlat();
		panelStyle.BgColor = new Color(0.03f, 0.05f, 0.08f, 0.84f);
		panelStyle.BorderColor = new Color(0.96f, 0.83f, 0.45f, 0.30f);
		panelStyle.SetBorderWidthAll(1);
		panelStyle.SetCornerRadiusAll(16);
		panelStyle.ShadowColor = new Color(0f, 0f, 0f, 0.36f);
		panelStyle.ShadowSize = 10;
		panel.AddThemeStyleboxOverride("panel", panelStyle);

		MarginContainer margin = new MarginContainer();
		margin.AddThemeConstantOverride("margin_left", 18);
		margin.AddThemeConstantOverride("margin_right", 18);
		margin.AddThemeConstantOverride("margin_top", 16);
		margin.AddThemeConstantOverride("margin_bottom", 16);
		panel.AddChild(margin);

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
	}

	public override void _Process(double delta)
	{
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
		sb.Append("[font_size=18][b][color=#ffe3a1]").Append(title).AppendLine("[/color][/b][/font_size]");
		sb.Append("[color=#a9c6de]Act ").Append(payload.act ?? 0)
			.Append("  |  Floor ").Append(payload.floor ?? 0);
		if (payload.step.HasValue)
		{
			sb.Append("  |  Step ").Append(payload.step.Value);
		}
		sb.Append("  |  ").Append(stateType).AppendLine("[/color]");
		sb.AppendLine();
		sb.Append("[font_size=22][b][color=#ffffff]").Append(chosenAction).AppendLine("[/color][/b][/font_size]");

		if (payload.options is { Length: > 0 })
		{
			sb.AppendLine("[color=#8fd3ff]Options[/color]");
			foreach (OverlayOptionPayload option in payload.options.Take(4))
			{
				bool chosen = ParseChosen(option.chosen);
				sb.Append(chosen ? "[color=#ffd978]> [/color]" : "[color=#55697d]- [/color]");
				sb.Append(string.IsNullOrWhiteSpace(option.label) ? "?" : option.label);
				if (option.cost.HasValue)
				{
					sb.Append("  [color=#9fc1df]c=").Append(option.cost.Value).Append("[/color]");
				}
				if (!string.IsNullOrWhiteSpace(option.target))
				{
					sb.Append("  [color=#9fc1df]-> ").Append(option.target).Append("[/color]");
				}
				sb.AppendLine();
			}
			sb.AppendLine();
		}

		if (payload.details is { Length: > 0 })
		{
			sb.AppendLine("[color=#8fd3ff]State[/color]");
			foreach (string detail in payload.details.Take(4))
			{
				if (!string.IsNullOrWhiteSpace(detail))
				{
					sb.Append("[color=#cad7e4]").Append(detail).AppendLine("[/color]");
				}
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
}
