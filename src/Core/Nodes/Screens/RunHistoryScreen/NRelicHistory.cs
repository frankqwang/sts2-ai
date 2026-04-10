using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Godot;
using MegaCrit.Sts2.addons.mega_text;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Exceptions;
using MegaCrit.Sts2.Core.Models.Relics;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Relics;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Nodes.Screens.RunHistoryScreen;

public partial class NRelicHistory : VBoxContainer
{
	[Signal]
	public delegate void HoveredEventHandler(NRelicBasicHolder relic);

	[Signal]
	public delegate void UnhoveredEventHandler(NRelicBasicHolder relic);

	private readonly LocString _relicHeader = new LocString("run_history", "RELIC_HISTORY.header");

	private readonly LocString _relicCategories = new LocString("run_history", "RELIC_HISTORY.categories");

	private MegaRichTextLabel _headerLabel;

	private Control _relicsContainer;

	public override void _Ready()
	{
		_headerLabel = GetNode<MegaRichTextLabel>("Header");
		_relicsContainer = GetNode<Control>("%RelicsContainer");
	}

	public void LoadRelics(Player player, IEnumerable<SerializableRelic> relics)
	{
		StringBuilder stringBuilder = new StringBuilder();
		foreach (Node child in _relicsContainer.GetChildren())
		{
			child.QueueFreeSafely();
		}
		Dictionary<RelicRarity, int> dictionary = new Dictionary<RelicRarity, int>();
		RelicRarity[] values = Enum.GetValues<RelicRarity>();
		foreach (RelicRarity key in values)
		{
			dictionary.Add(key, 0);
		}
		List<SerializableRelic> list = relics.ToList();
		foreach (SerializableRelic item in list)
		{
			RelicModel relicModel;
			try
			{
				relicModel = RelicModel.FromSerializable(item);
			}
			catch (ModelNotFoundException)
			{
				relicModel = ModelDb.Relic<DeprecatedRelic>().ToMutable();
			}
			relicModel.Owner = player;
			NRelicBasicHolder holder = NRelicBasicHolder.Create(relicModel);
			holder.MouseDefaultCursorShape = CursorShape.Help;
			_relicsContainer.AddChildSafely(holder);
			holder.Connect(Control.SignalName.FocusEntered, Callable.From(delegate
			{
				EmitSignal(SignalName.Hovered, holder);
			}));
			holder.Connect(Control.SignalName.FocusExited, Callable.From(delegate
			{
				EmitSignal(SignalName.Unhovered, holder);
			}));
			holder.Connect(Control.SignalName.MouseEntered, Callable.From(delegate
			{
				EmitSignal(SignalName.Hovered, holder);
			}));
			holder.Connect(Control.SignalName.MouseExited, Callable.From(delegate
			{
				EmitSignal(SignalName.Unhovered, holder);
			}));
			holder.Connect(NClickableControl.SignalName.Released, Callable.From<NButton>(delegate
			{
				OnRelicClicked(holder.Relic);
			}));
			dictionary[relicModel.Rarity]++;
		}
		_relicHeader.Add("totalRelics", list.Count);
		foreach (KeyValuePair<RelicRarity, int> item2 in dictionary)
		{
			_relicCategories.Add(item2.Key.ToString() + "Relics", item2.Value);
		}
		StringBuilder stringBuilder2 = stringBuilder;
		StringBuilder.AppendInterpolatedStringHandler handler = new StringBuilder.AppendInterpolatedStringHandler(20, 1, stringBuilder2);
		handler.AppendLiteral("[gold][b]");
		handler.AppendFormatted(_relicHeader.GetFormattedText());
		handler.AppendLiteral("[/b][/gold]");
		stringBuilder2.Append(ref handler);
		stringBuilder.Append(_relicCategories.GetFormattedText().Trim(','));
		_headerLabel.Text = stringBuilder.ToString();
	}

	private void OnRelicClicked(NRelic node)
	{
		List<RelicModel> list = new List<RelicModel>();
		foreach (NRelicBasicHolder item in _relicsContainer.GetChildren().OfType<NRelicBasicHolder>())
		{
			list.Add(item.Relic.Model);
		}
		NGame.Instance.GetInspectRelicScreen().Open(list, node.Model);
	}
}
