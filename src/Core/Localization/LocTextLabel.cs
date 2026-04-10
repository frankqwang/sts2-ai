using System;
using Godot;

namespace MegaCrit.Sts2.Core.Localization;

public partial class LocTextLabel : RichTextLabel
{
	[Export(PropertyHint.None, "")]
	private string? _localizationTable;

	[Export(PropertyHint.None, "")]
	private string? _localizationKey;

	private LocString? _locString;

	public string? LocalizationTable
	{
		get
		{
			return _localizationTable;
		}
		set
		{
			if (!(_localizationTable == value))
			{
				_localizationTable = value;
				_locString = null;
				UpdateLocalization();
			}
		}
	}

	public string? LocalizationKey
	{
		get
		{
			return _localizationKey;
		}
		set
		{
			if (!(_localizationKey == value))
			{
				_localizationKey = value;
				_locString = null;
				UpdateLocalization();
			}
		}
	}

	private void UpdateLocalization()
	{
		if (_localizationTable == null)
		{
			throw new InvalidOperationException("_localizationTable is null.");
		}
		if (_localizationKey == null)
		{
			throw new InvalidOperationException("_localizationKey is null.");
		}
		if (_locString == null)
		{
			_locString = new LocString(_localizationTable, _localizationKey);
		}
		base.Text = _locString.GetFormattedText();
	}

	public override void _Ready()
	{
		UpdateLocalization();
	}
}
