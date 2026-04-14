$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$configPath = Join-Path $repoRoot "STS2AI\Python\configs\hybrid_train_ironclad_teacher_main_attention_mcts.toml"
python "$repoRoot\STS2AI\Python\train_hybrid.py" --config $configPath @args
