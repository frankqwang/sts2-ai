using System.Linq;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.Logging;
using MegaCrit.Sts2.Core.Saves;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace MegaCrit.Sts2.Core.Models.Badges;

public abstract class Badge
{
	protected readonly SerializableRun _run;

	protected readonly SerializablePlayer _localPlayer;

	public virtual string Id => "NOT_SET";

	public virtual BadgeRarity Rarity => BadgeRarity.None;

	public abstract bool RequiresWin { get; }

	public abstract bool MultiplayerOnly { get; }

	public Texture2D BadgeBase => Rarity switch
	{
		BadgeRarity.Bronze => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_bronze.png")), 
		BadgeRarity.Silver => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_silver.png")), 
		BadgeRarity.Gold => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_gold.png")), 
		_ => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("atlases/power_atlas.sprites/missing_power.tres")), 
	};

	public Texture2D BadgeIcon
	{
		get
		{
			if (ResourceLoader.Exists(IconPath))
			{
				return PreloadManager.Cache.GetTexture2D(IconPath);
			}
			Log.Error("Badge Icon: " + IconPath + " doesn't exist :(");
			return PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("debug/placeholder_64.png"));
		}
	}

	private string IconPath => ImageHelper.GetImagePath("ui/game_over_screen/badge_" + Id.ToLowerInvariant() + ".png");

	protected Badge(SerializableRun run, ulong playerId)
	{
		_run = run;
		_localPlayer = _run.Players.First((SerializablePlayer p) => p.NetId == playerId);
	}

	public abstract bool IsObtained();

	public SerializableBadge ToSerializable()
	{
		return new SerializableBadge
		{
			Id = Id,
			Rarity = Rarity
		};
	}
}
