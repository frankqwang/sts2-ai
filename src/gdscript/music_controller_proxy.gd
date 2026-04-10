
class_name MusicControllerProxy extends Node

var _musicEv: FmodEvent
var _currentTrack

var _ambienceEv: FmodEvent
var _currentAmbience

var _bank_loader: FmodBankLoader

func update_music(track):
    _currentTrack = track
    stop_music()

    if !FmodServer.check_event_path(_currentTrack):
        printerr("cannot find music path: " + _currentTrack)
        return

    _musicEv = FmodServer.create_event_instance(_currentTrack)
    _musicEv.start()

func update_music_parameter(label, labelIndex):
    if _musicEv == null:
        printerr("missing music track: " + _currentTrack)
        return

    _musicEv.set_parameter_by_name(label, labelIndex)

func update_global_parameter(label, labelIndex):
    FmodServer.set_global_parameter_by_name(label, labelIndex)

func stop_music():
    if _musicEv != null:
        _musicEv.stop(0)
        _musicEv.release()
        _musicEv = null

func update_ambience(track):
    _currentAmbience = track
    stop_ambience()

    if !FmodServer.check_event_path(_currentAmbience):
        printerr("cannot find ambience path: " + _currentAmbience)
        return

    _ambienceEv = FmodServer.create_event_instance(_currentAmbience)
    _ambienceEv.start()

func update_campfire_ambience(trackIndex):
    if !FmodServer.check_event_path(_currentAmbience):
        printerr("cannot find ambience path: " + _currentAmbience)
        return

    _ambienceEv.set_parameter_by_name("Campfire", trackIndex)

func stop_ambience():
    if _ambienceEv != null:
        _ambienceEv.stop(0)
        _ambienceEv.release()
        _ambienceEv = null

func load_act_banks(bank_paths: Array):
    if _bank_loader and _bank_loader.bank_paths == bank_paths:
        return
    unload_act_banks()
    for path in bank_paths:
        if not FileAccess.file_exists(path):
            printerr("music bank not found: " + path)
            return
    _bank_loader = FmodBankLoader.new()
    _bank_loader.bank_paths = bank_paths
    add_child(_bank_loader)

func unload_act_banks():
    if _bank_loader:
        remove_child(_bank_loader)








        _bank_loader.free()
        _bank_loader = null
