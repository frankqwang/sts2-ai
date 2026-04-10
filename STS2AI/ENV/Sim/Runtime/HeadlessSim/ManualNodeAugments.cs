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

        public void SetVisible(bool visible) { }

        public static NCreature? Create(Creature entity, PotionModel? inspectPotion = null) => default;
    }
}
