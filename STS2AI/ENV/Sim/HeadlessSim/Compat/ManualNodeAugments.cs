using System.Collections.Generic;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Nodes.RestSite;
using MegaCrit.Sts2.Core.Training;

namespace MegaCrit.Sts2.Core.Nodes.Rooms
{
    public partial class NRestSiteRoom
    {
        public List<NRestSiteCharacter> characterAnims { get; } = new();
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Combat
{
    public partial class NCreature
    {
        public Vector2 Scale { get; set; } = Vector2.One;

        public MegaCrit.Sts2.Core.Bindings.MegaSpine.MegaSprite? SpineController { get; set; }

        public void SetVisible(bool visible) { }

        public static NCreature? Create(Creature entity, PotionModel? inspectPotion = null) => default;
    }

    public partial class NSelectedHandCardContainer : Control
    {
        public List<Cards.Holders.NSelectedHandCardHolder> Holders { get; } = new();
    }
}

namespace MegaCrit.Sts2.Core.Nodes.CommonUi
{
    public partial class NConfirmButton : MegaCrit.Sts2.Core.Nodes.GodotExtensions.NButton
    {
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Events
{
    public partial class NAncientEventLayout : NEventLayout
    {
        public bool TryAdvanceDialogue() => false;
    }
}

namespace MegaCrit.Sts2.Core.Nodes.Screens
{
    public partial class NRewardsScreen
    {
        public bool IsConnected(StringName signal, Callable callable) => false;
    }
}
