using System;
using Godot;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Characters;
using MegaCrit.Sts2.Core.Nodes.Screens.Settings;

namespace MegaCrit.Sts2.Core.Nodes.Debug.Multiplayer;

public partial class NMultiplayerTestCharacterPaginator : NPaginator
{
	private readonly CharacterModel[] _characters = new CharacterModel[5]
	{
		ModelDb.Character<Ironclad>(),
		ModelDb.Character<Silent>(),
		ModelDb.Character<Regent>(),
		ModelDb.Character<Necrobinder>(),
		ModelDb.Character<Defect>()
	};

	public CharacterModel Character => _characters[_currentIndex];

	public event Action<CharacterModel>? CharacterChanged;

	public override void _Ready()
	{
		ConnectSignals();
		CharacterModel[] characters = _characters;
		foreach (CharacterModel characterModel in characters)
		{
			_options.Add(new LocString("characters", characterModel.Id.Entry + ".title").GetFormattedText());
		}
		_label.Text = _options[_currentIndex];
	}

	protected override void OnIndexChanged(int index)
	{
		_currentIndex = index;
		_label.Text = _characters[index].Title.GetFormattedText();
		this.CharacterChanged?.Invoke(_characters[index]);
	}
}
