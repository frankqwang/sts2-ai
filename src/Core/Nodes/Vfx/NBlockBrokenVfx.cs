using System.Collections.Generic;
using Godot;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Vfx;

public partial class NBlockBrokenVfx : Sprite2D
{
	private const string _scenePath = "res://scenes/vfx/vfx_block_broken.tscn";

	public static IEnumerable<string> AssetPaths => new global::_003C_003Ez__ReadOnlySingleElementList<string>("res://scenes/vfx/vfx_block_broken.tscn");

	public override void _Ready()
	{
		AnimationPlayer node = GetNode<AnimationPlayer>("AnimationPlayer");
		node.Connect(AnimationMixer.SignalName.AnimationFinished, Callable.From<StringName>(OnAnimationFinished));
	}

	private void OnAnimationFinished(StringName _)
	{
		this.QueueFreeSafely();
	}

	public static NBlockBrokenVfx? Create()
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		return PreloadManager.Cache.GetScene("res://scenes/vfx/vfx_block_broken.tscn").Instantiate<NBlockBrokenVfx>(PackedScene.GenEditState.Disabled);
	}
}
