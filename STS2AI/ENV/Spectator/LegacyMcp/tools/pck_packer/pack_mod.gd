extends SceneTree

func _init() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 2:
		push_error("Usage: godot --headless --path <packer_project> --script pack_mod.gd -- <manifest_src> <out_pck>")
		quit(1)
		return

	var manifest_src := args[0]
	var out_pck := args[1]
	var packer := PCKPacker.new()

	var err := packer.pck_start(out_pck)
	if err != OK:
		push_error("Failed to start PCK '%s' (error %d)" % [out_pck, err])
		quit(err)
		return

	err = packer.add_file("res://mod_manifest.json", manifest_src)
	if err != OK:
		push_error("Failed to add mod_manifest.json from '%s' (error %d)" % [manifest_src, err])
		quit(err)
		return

	err = packer.flush()
	if err != OK:
		push_error("Failed to flush PCK '%s' (error %d)" % [out_pck, err])

	quit(err)
